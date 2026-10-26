// SafeCell AI - on-chip battery health (SoH) + overheating alarm for an ESP32 BMS.
//
// The 4 test cycles in test_cycles.h are streamed one sample at a time, like
// live sensor readings. For every sample we:
//   1. run the overheating alarm (dT/dt threshold + 60 C hard limit)
//   2. update the 4 window features
// When the voltage window (3.8 V -> 3.5 V) ends, we run the INT8 SoH model.

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "constants.h"
#include "test_cycles.h"

const int LED_GREEN = 25;   // normal
const int LED_RED = 26;     // alarm
const int SAMPLE_DELAY_MS = 5;  // pause between samples so the stream looks "live"

// ---------- TensorFlow Lite Micro ----------
const int kArenaSize = 4 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ---------- Per-cycle state ----------
struct FeatureState {
  float v0;                 // voltage of first sample
  bool has_v60;             // v_drop_60s computed?
  float v_drop_60s;
  bool in_window;           // passed 3.8 V
  float t_start, temp_start;
  bool done;                // passed 3.5 V, features ready
  float dt_window, temp_rise;
};

struct AlarmState {
  float prev_t, prev_temp;
  float rates[3];           // last 3 dT/dt values (ring buffer)
  int n_rates;              // how many rates seen so far
  int above;                // consecutive samples above threshold
  bool alarm;               // latched for the rest of the cycle
};

void setLeds(bool alarm) {
  digitalWrite(LED_GREEN, alarm ? LOW : HIGH);
  digitalWrite(LED_RED, alarm ? HIGH : LOW);
}

// Overheating alarm for one sample. Same logic as run_alarm() in src/alarm.py.
// k = sample index in the cycle. Returns true on the sample the alarm first fires.
bool alarmStep(AlarmState& a, int k, float t, float temp) {
  if (a.alarm) return false;  // already latched
  if (k > 0) {
    a.rates[a.n_rates % 3] = (temp - a.prev_temp) / (t - a.prev_t);
    a.n_rates++;
  }
  a.prev_t = t;
  a.prev_temp = temp;

  if (temp > TEMP_LIMIT_C) {
    a.alarm = true;
    Serial.printf("  ALARM at t=%.1f s: temperature %.1f C > %.0f C limit\n", t, temp, TEMP_LIMIT_C);
    return true;
  }
  if (k >= 3) {  // need 3 rates for the moving average
    float ma = (a.rates[0] + a.rates[1] + a.rates[2]) / 3.0f;
    if (ma > DTDT_THRESHOLD) {
      a.above++;
      if (a.above >= ALARM_CONSECUTIVE) {
        a.alarm = true;
        Serial.printf("  ALARM at t=%.1f s: dT/dt=%.4f C/s > %.4f C/s for %d samples\n",
                      t, ma, DTDT_THRESHOLD, ALARM_CONSECUTIVE);
        return true;
      }
    } else {
      a.above = 0;
    }
  }
  return false;
}

// Window features for one sample. Same logic as compute_features() in src/data.py.
// Returns true on the sample where all 4 features become ready.
bool featureStep(FeatureState& f, int k, float t, float v, float temp) {
  if (f.done) return false;
  if (k == 0) f.v0 = v;
  if (!f.has_v60 && t >= 60.0f) {
    f.v_drop_60s = f.v0 - v;
    f.has_v60 = true;
  }
  if (!f.in_window && v <= V_WINDOW_START) {
    f.in_window = true;
    f.t_start = t;
    f.temp_start = temp;
  }
  if (f.in_window && v <= V_WINDOW_END) {
    f.dt_window = t - f.t_start;
    f.temp_rise = temp - f.temp_start;
    f.done = true;
  }
  return f.done && f.has_v60;
}

// Run the INT8 model on the 4 features. Returns SoH in %.
float predictSoh(const FeatureState& f, unsigned long* micros_taken) {
  float x[4] = {f.dt_window, f.temp_start, f.temp_rise, f.v_drop_60s};
  float in_scale = input->params.scale;
  int in_zero = input->params.zero_point;
  for (int j = 0; j < 4; j++) {
    float scaled = (x[j] - FEATURE_MEAN[j]) / FEATURE_SCALE[j];  // standard scaler
    int q = (int)roundf(scaled / in_scale) + in_zero;             // float -> int8
    input->data.int8[j] = (int8_t)constrain(q, -128, 127);
  }
  unsigned long t0 = micros();
  interpreter->Invoke();
  *micros_taken = micros() - t0;
  int8_t q_out = output->data.int8[0];
  return (q_out - output->params.zero_point) * output->params.scale * 100.0f;  // int8 -> SoH %
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  setLeds(false);
  delay(500);
  Serial.println("\n=== SafeCell AI: on-chip SoH + overheating alarm ===");

  const tflite::Model* model = tflite::GetModel(g_model);
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, &error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (true) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  unsigned long total_us = 0;
  int n_inferences = 0;

  for (int c = 0; c < NUM_TEST_CYCLES; c++) {
    const TestCycle& tc = TEST_CYCLES[c];
    FeatureState f = {};
    AlarmState a = {};
    setLeds(false);
    Serial.printf("\n--- Cycle %d (%s, dataset cycle %d, %d samples) ---\n", c + 1, tc.label, tc.cycle, tc.n);

    for (int k = 0; k < tc.n; k++) {
      // One "live" sensor reading
      float t = tc.t[k], v = tc.v[k], temp = tc.temp[k];

      if (alarmStep(a, k, t, temp)) setLeds(true);

      if (featureStep(f, k, t, v, temp)) {
        unsigned long us = 0;
        float soh = predictSoh(f, &us);
        total_us += us;
        n_inferences++;
        Serial.printf("  features: dt_window=%.3f temp_start=%.3f temp_rise=%.3f v_drop_60s=%.4f\n",
                      f.dt_window, f.temp_start, f.temp_rise, f.v_drop_60s);
        Serial.printf("Cycle %d: SoH predicted %.1f%% | true %.1f%% | inference %lu us\n",
                      c + 1, soh, tc.true_soh, us);
      }
      delay(SAMPLE_DELAY_MS);
    }
    Serial.printf("  end of cycle: %s\n", a.alarm ? "ALARM (red LED)" : "normal (green LED)");
    delay(1000);
  }

  Serial.println("\n=== Summary (timings measured in Wokwi simulation) ===");
  Serial.printf("Model size: %u bytes (INT8)\n", g_model_len);
  Serial.printf("Tensor arena used: %u of %d bytes\n", (unsigned)interpreter->arena_used_bytes(), kArenaSize);
  Serial.printf("Average inference time: %lu us over %d inferences\n",
                n_inferences ? total_us / n_inferences : 0, n_inferences);
  Serial.println("Done.");
  while (true) delay(1000);  // run once
}

// ============================================================================
//  Physical-Modeling Piano Synthesizer  (Yamaha C5 style demo)
//  ---------------------------------------------------------------------------
//  Version: 2.0.0      (previous: 1.0.0  = 初版 Karplus-Strong demo)
//
//  v2.0.0 相对 v1.0.0 的改进:
//    * 锤击激励: 速度相关的脉冲宽度(硬触键更短更亮) + 毛毡噪声瞬态
//    * 色散全通链(dispersion allpass cascade): 模拟弦的刚度/非谐性,
//      高次分音被上拉(stretched partials), 摆脱"塑料感"
//    * 双段衰减(prompt + after sound): 起音段衰减快, 尾音段衰减慢
//    * 同音弦按 cents 精确失谐: 产生真实的拍频与厚度
//    * 立体声: 按音高分布声像(低音偏左/高音偏右) + 立体声音板共鸣
//    * 阻尼器噪声(松键瞬间) + 轻微板式混响: 空间感与"呼吸感"
//    * 声板共鸣器扩展到 6 模态立体声 + 高频滚降
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <algorithm>

#define SYNTH_VERSION            "2.0.0"

#define SAMPLE_RATE              44100
#define AUDIO_BLOCK_SIZE         64
#define MAX_DELAY_LEN            1200
#define NUM_AP                   4       // 每条弦的色散全通级数
#define VOICE_COUNT              24
#define PRECOMP_KEY_COUNT        61
#define VOICE_RELEASE_SAMPLES    384

const float PI_F = 3.14159265358979323846f;
const float VOICE_MIX_SCALE = 0.30f;
const float SOFT_CLIP_T = 0.82f;
const float DC_BLOCK_R = 0.9957f;
const float AP_COEFF = -0.30f;             // 色散全通系数 (负值 -> 高次分音上拉)
const float STRING_CROSS = 0.006f;         // 同音弦桥耦合量

const float note_frequencies[PRECOMP_KEY_COUNT] = {
    65.41f,  69.30f,  73.42f,  77.78f,  82.41f,  87.31f,  92.50f,  98.00f, 103.83f, 110.00f, 116.54f, 123.47f,
   130.81f, 138.59f, 146.83f, 155.56f, 164.81f, 174.61f, 185.00f, 196.00f, 207.65f, 220.00f, 233.08f, 246.94f,
   261.63f, 277.18f, 293.66f, 311.13f, 329.63f, 349.23f, 369.99f, 392.00f, 415.30f, 440.00f, 466.16f, 493.88f,
   523.25f, 554.37f, 587.33f, 622.25f, 659.25f, 698.46f, 739.99f, 783.99f, 830.61f, 880.00f, 932.33f, 987.77f,
  1046.50f,1108.73f,1174.66f,1244.51f,1318.51f,1396.91f,1479.98f,1567.98f,1661.22f,1760.00f,1864.66f,1975.53f,
  2093.00f
};

uint16_t master_gain_q15 = 26000;
uint8_t global_sustain_pedal = 0;
static float master_gain_smooth = 0.0f;
static bool gain_smooth_initialized = false;
static float block_mix_L[AUDIO_BLOCK_SIZE];
static float block_mix_R[AUDIO_BLOCK_SIZE];

// ---------------------------------------------------------------------------
//  工具函数
// ---------------------------------------------------------------------------
inline float clampFloat(float value, float minimum, float maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

inline float softClip(float x) {
  const float ax = std::abs(x);
  if (ax <= SOFT_CLIP_T) return x;
  const float sign = (x >= 0.0f) ? 1.0f : -1.0f;
  const float d = ax - SOFT_CLIP_T;
  const float range = 1.0f - SOFT_CLIP_T;
  return sign * (SOFT_CLIP_T + range * (d / (d + range)));
}

inline float whiteNoise() {
  static uint32_t seed = 0x1234567u;
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return (float)(int32_t)seed * (1.0f / 2147483648.0f);
}

// 一阶全通在给定角频率处的群延迟 (用于把色散延迟从弦长里扣除)
inline float allpassGroupDelay(float a, float w) {
  if (w < 1e-6f) return (1.0f - a) / (1.0f + a);
  float c = std::cos(w);
  return (1.0f - a * a) / (1.0f + 2.0f * a * c + a * a);
}

// 一阶低通在给定角频率处的相位延迟
inline float onepolePhaseDelay(float a, float w) {
  if (w < 1e-6f) return a / (1.0f - a);
  float s = std::sin(w);
  float c = std::cos(w);
  return std::atan2(a * s, 1.0f - a * c) / w;
}

// ---------------------------------------------------------------------------
//  基础滤波器
// ---------------------------------------------------------------------------
struct Biquad {
  float b0, b1, b2, a1, a2;
  float s1, s2;
  inline void reset() { s1 = 0.0f; s2 = 0.0f; }
  inline float process(float in) {
    float out = b0 * in + s1;
    s1 = b1 * in - a1 * out + s2;
    s2 = b2 * in - a2 * out;
    if (std::abs(s1) < 1e-15f) s1 = 0.0f;
    if (std::abs(s2) < 1e-15f) s2 = 0.0f;
    return out;
  }
};

struct Allpass1 {
  float a;
  float state;
  inline void reset() { state = 0.0f; }
  inline float process(float in) {
    float out = a * in + state;
    state = in - a * out;
    if (std::abs(state) < 1e-15f) state = 0.0f;
    return out;
  }
};

struct OnePoleLP {
  float a;
  float z;
  inline void reset() { z = 0.0f; }
  inline float process(float in) {
    z = (1.0f - a) * in + a * z;
    if (std::abs(z) < 1e-15f) z = 0.0f;
    return z;
  }
};

void calcLowpass(Biquad& bq, float fc, float fs) {
  float w0 = 2.0f * PI_F * fc / fs;
  float cosw0 = std::cos(w0);
  float sinw0 = std::sin(w0);
  float alpha = sinw0 / (2.0f * 0.707f);
  float a0 = 1.0f + alpha;
  bq.b0 = ((1.0f - cosw0) / 2.0f) / a0;
  bq.b1 = (1.0f - cosw0) / a0;
  bq.b2 = bq.b0;
  bq.a1 = (-2.0f * cosw0) / a0;
  bq.a2 = (1.0f - alpha) / a0;
  bq.reset();
}

void calcBandpass(Biquad& bq, float fc, float Q, float fs) {
  float w0 = 2.0f * PI_F * fc / fs;
  float cosw0 = std::cos(w0);
  float sinw0 = std::sin(w0);
  float alpha = sinw0 / (2.0f * Q);
  float a0 = 1.0f + alpha;
  bq.b0 = alpha / a0;
  bq.b1 = 0.0f;
  bq.b2 = -alpha / a0;
  bq.a1 = (-2.0f * cosw0) / a0;
  bq.a2 = (1.0f - alpha) / a0;
  bq.reset();
}

// ---------------------------------------------------------------------------
//  立体声共鸣声板 (6 模态/声道 + 高频滚降)
// ---------------------------------------------------------------------------
struct Soundboard {
  Biquad modeL[6], modeR[6];
  Biquad lpL, lpR;
  float wL[6], wR[6];

  void init(float fs) {
    float fL[6] = { 95.0f, 190.0f, 430.0f,  900.0f, 1900.0f, 3800.0f };
    float fR[6] = { 88.0f, 205.0f, 470.0f,  980.0f, 2050.0f, 4200.0f };
    float Q[6]  = { 1.3f,  1.6f,   2.0f,    2.4f,    2.9f,    3.4f };
    float wgt[6]= { 0.30f, 0.26f,  0.20f,   0.14f,   0.08f,   0.05f };
    for (int i = 0; i < 6; i++) {
      calcBandpass(modeL[i], fL[i], Q[i], fs);
      calcBandpass(modeR[i], fR[i], Q[i], fs);
      wL[i] = wgt[i];
      wR[i] = wgt[i];
    }
    calcLowpass(lpL, 11000.0f, fs);
    calcLowpass(lpR, 10500.0f, fs);
  }

  inline float processL(float in) {
    float s = 0.0f;
    for (int i = 0; i < 6; i++) s += modeL[i].process(in) * wL[i];
    return lpL.process(in * 0.55f + s);
  }
  inline float processR(float in) {
    float s = 0.0f;
    for (int i = 0; i < 6; i++) s += modeR[i].process(in) * wR[i];
    return lpR.process(in * 0.55f + s);
  }
};

// ---------------------------------------------------------------------------
//  紧凑板式混响 (Schroeder: 4 comb + 2 allpass 每声道)
// ---------------------------------------------------------------------------
struct DampComb {
  std::vector<float> buf;
  int idx;
  float fb, damp, store;
  DampComb() : idx(0), fb(0.84f), damp(0.25f), store(0.0f) {}
  void init(int n) { buf.assign(n, 0.0f); idx = 0; store = 0.0f; }
  inline float process(float in) {
    float y = buf[idx];
    store = y * (1.0f - damp) + store * damp;
    buf[idx] = in + store * fb;
    if (++idx >= (int)buf.size()) idx = 0;
    return y;
  }
};

struct ReverbAP {
  std::vector<float> buf;
  int idx;
  float g;
  ReverbAP() : idx(0), g(0.5f) {}
  void init(int n) { buf.assign(n, 0.0f); idx = 0; }
  inline float process(float in) {
    float y = buf[idx];
    float out = -in + y;
    buf[idx] = in + y * g;
    if (++idx >= (int)buf.size()) idx = 0;
    return out;
  }
};

struct Reverb {
  DampComb cl[4], cr[4];
  ReverbAP al[2], ar[2];
  float wet;
  Reverb() : wet(0.16f) {}
  void init() {
    int ct[4] = { 1116, 1188, 1277, 1356 };
    int at[2] = { 556, 441 };
    for (int i = 0; i < 4; i++) { cl[i].init(ct[i]); cr[i].init(ct[i] + 23); }
    for (int i = 0; i < 2; i++) { al[i].init(at[i]); ar[i].init(at[i] + 23); }
  }
  inline void process(float inL, float inR, float& oL, float& oR) {
    float l = 0.0f, r = 0.0f;
    for (int i = 0; i < 4; i++) { l += cl[i].process(inL); r += cr[i].process(inR); }
    l *= 0.25f; r *= 0.25f;
    for (int i = 0; i < 2; i++) { l = al[i].process(l); r = ar[i].process(r); }
    oL = l * wet;
    oR = r * wet;
  }
};

Soundboard soundboard;
Reverb reverb;

// ---------------------------------------------------------------------------
//  预计算音符表
// ---------------------------------------------------------------------------
struct PrecalculatedNote {
  float freq;
  uint8_t numStrings;
  float detuneCents[3];
};

PrecalculatedNote precomp_notes[PRECOMP_KEY_COUNT];

void precomputeNoteTables() {
  for (int i = 0; i < PRECOMP_KEY_COUNT; i++) {
    float baseFreq = note_frequencies[i];
    uint8_t strings = 1;
    if (i >= 8 && i < 24) strings = 2;
    else if (i >= 24) strings = 3;

    precomp_notes[i].freq = baseFreq;
    precomp_notes[i].numStrings = strings;

    // 高音区失谐更大 (真实钢琴钢丝更硬, 失谐随音高增大)
    float scale = 0.8f + 1.0f * ((float)i / 60.0f);
    if (strings == 1) {
      precomp_notes[i].detuneCents[0] = 0.0f;
      precomp_notes[i].detuneCents[1] = 0.0f;
      precomp_notes[i].detuneCents[2] = 0.0f;
    } else if (strings == 2) {
      precomp_notes[i].detuneCents[0] = -0.6f * scale;
      precomp_notes[i].detuneCents[1] =  0.6f * scale;
      precomp_notes[i].detuneCents[2] =  0.0f;
    } else {
      precomp_notes[i].detuneCents[0] = -0.9f * scale;
      precomp_notes[i].detuneCents[1] =  0.0f;
      precomp_notes[i].detuneCents[2] =  0.9f * scale;
    }
  }
}

// ---------------------------------------------------------------------------
//  钢琴声部 (每条弦 = 延迟线 + 环路阻尼低通 + 色散全通链)
// ---------------------------------------------------------------------------
struct PianoVoice {
  float delay[3][MAX_DELAY_LEN];
  int writeIdx[3];
  float delayLen[3];
  OnePoleLP loopLP[3];
  Allpass1 disp[3][NUM_AP];
  float feedback[3];
  float gFast[3], gSlow[3], gDamp[3];

  uint8_t numStrings;
  int injectCount, injectLen;
  float injectAmp, noiseAmp, noiseDecay;

  float panL, panR;
  float velNorm;

  int damperCount;
  bool wasSustained;

  bool active;
  uint32_t age;
  uint32_t ageSamples;
  int noteIndex;
  uint8_t pedal;
  uint16_t silentCount;
  float gain;
  uint16_t fadeCount;
  bool hasPending;
  int8_t pendNote;
  uint8_t pendVel;
  uint8_t pendPedal;
};

PianoVoice voices[VOICE_COUNT];
uint32_t voice_age_counter = 0;
static float dc_prev_in_L = 0.0f, dc_prev_out_L = 0.0f;
static float dc_prev_in_R = 0.0f, dc_prev_out_R = 0.0f;

void startVoice(PianoVoice& v, int noteIndex, uint8_t velocity, uint8_t pedal) {
  const float fs = (float)SAMPLE_RATE;
  const PrecalculatedNote& pn = precomp_notes[noteIndex];

  v.noteIndex = noteIndex;
  v.pedal = pedal;
  v.active = true;
  v.silentCount = 0;
  v.gain = 1.0f;
  v.fadeCount = 0;
  v.hasPending = false;
  v.damperCount = 0;
  v.ageSamples = 0;
  voice_age_counter++;
  v.age = voice_age_counter;

  float f0 = pn.freq;
  float velNorm = (float)velocity / 127.0f;
  if (velNorm < 0.02f) velNorm = 0.02f;
  v.velNorm = velNorm;
  float freqFactor = clampFloat(f0 / 2093.0f, 0.0f, 1.0f);

  // 声像: 低音偏左, 高音偏右 (等功率)
  float pan = clampFloat(((float)noteIndex / 60.0f) * 1.4f - 0.7f, -0.7f, 0.7f);
  float th = (pan + 1.0f) * 0.25f * PI_F;
  v.panL = std::cos(th);
  v.panR = std::sin(th);

  // 环路阻尼低通: 控制高次分音衰减速率; 触键越硬截止越高 -> 越亮
  float fc = clampFloat(f0 * 3.2f + 1500.0f, 1500.0f, 8000.0f);
  fc *= (0.8f + 0.5f * velNorm);
  if (fc > fs * 0.45f) fc = fs * 0.45f;
  float lpCoeff = std::exp(-2.0f * PI_F * fc / fs);

  float w0 = 2.0f * PI_F * f0 / fs;
  float dispDelay = NUM_AP * allpassGroupDelay(AP_COEFF, w0);
  float lpDelay = onepolePhaseDelay(lpCoeff, w0);

  // 衰减时间 (双段: prompt sound + after sound)
  float t60 = 9.5f * std::pow(65.41f / f0, 0.72f);
  if (t60 > 18.0f) t60 = 18.0f;
  float t60Fast = t60 * 0.30f;
  float t60Damp = 0.30f - 0.18f * freqFactor;
  if (t60Damp < 0.10f) t60Damp = 0.10f;

  float totalLoop = fs / f0;

  v.numStrings = pn.numStrings;
  v.injectCount = 0;
  float pulseLen = clampFloat(totalLoop * 0.5f * (1.2f - 0.7f * velNorm), 6.0f, 90.0f);
  v.injectLen = (int)pulseLen;
  v.injectAmp = 0.85f * std::pow(velNorm, 1.25f);
  v.noiseAmp = 0.05f * velNorm * (1.4f - 0.8f * freqFactor);
  v.noiseDecay = 0.0025f * fs * (1.0f + 2.0f * freqFactor);
  v.wasSustained = (global_sustain_pedal > 0) || (pedal > 0);

  for (int s = 0; s < 3; s++) {
    for (int k = 0; k < MAX_DELAY_LEN; k++) v.delay[s][k] = 0.0f;
    v.writeIdx[s] = 0;
    v.delayLen[s] = 4.0f;
    v.feedback[s] = 0.0f;
    v.loopLP[s].a = lpCoeff;
    v.loopLP[s].reset();
    for (int m = 0; m < NUM_AP; m++) { v.disp[s][m].a = AP_COEFF; v.disp[s][m].reset(); }
    v.gFast[s] = 0.999f; v.gSlow[s] = 0.999f; v.gDamp[s] = 0.98f;
  }

  for (int s = 0; s < v.numStrings; s++) {
    float ratio = std::pow(2.0f, pn.detuneCents[s] / 1200.0f);
    float loopS = totalLoop / ratio;
    float dLen = loopS - dispDelay - lpDelay;
    if (dLen < 4.0f) dLen = 4.0f;
    v.delayLen[s] = dLen;

    float var = 1.0f + 0.01f * (float)(s - 1);
    v.gSlow[s] = std::exp(-6.9078f * loopS / (fs * t60 * var));
    v.gFast[s] = std::exp(-6.9078f * loopS / (fs * t60Fast * var));
    v.gDamp[s] = std::exp(-6.9078f * loopS / (fs * t60Damp * var));
    v.feedback[s] = v.gFast[s];   // 起音段使用快速衰减
  }
}

void triggerPianoNote(int noteIndex, uint8_t velocity, uint8_t pedal) {
  if (noteIndex < 0 || noteIndex >= PRECOMP_KEY_COUNT) return;
  if (velocity == 0) velocity = 80;

  int idleIdx = -1, oldestIdx = -1;
  uint32_t oldestAge = 0xFFFFFFFFu;

  for (int v = 0; v < VOICE_COUNT; v++) {
    if (!voices[v].active) { idleIdx = v; break; }
    if (voices[v].age < oldestAge) { oldestAge = voices[v].age; oldestIdx = v; }
  }

  if (idleIdx >= 0) {
    startVoice(voices[idleIdx], noteIndex, velocity, pedal);
    return;
  }

  PianoVoice& victim = voices[oldestIdx];
  if (!victim.fadeCount) victim.fadeCount = VOICE_RELEASE_SAMPLES;
  victim.hasPending = true;
  victim.pendNote   = (int8_t)noteIndex;
  victim.pendVel   = velocity;
  victim.pendPedal = pedal;
}

// ---------------------------------------------------------------------------
//  渲染单个 Block (64 采样点)
// ---------------------------------------------------------------------------
void renderBlock(int16_t* pcm_out) {
  for (int i = 0; i < AUDIO_BLOCK_SIZE; i++) { block_mix_L[i] = 0.0f; block_mix_R[i] = 0.0f; }

  const float fs = (float)SAMPLE_RATE;

  for (int v = 0; v < VOICE_COUNT; v++) {
    PianoVoice& voice = voices[v];
    if (!voice.active) continue;

    bool is_sustained = (global_sustain_pedal > 0) || (voice.pedal > 0);

    // 松键瞬间触发阻尼器噪声
    if (voice.wasSustained && !is_sustained) voice.damperCount = (int)(0.02f * fs);
    voice.wasSustained = is_sustained;

    // block 级计算双段衰减目标
    float t = (float)voice.ageSamples / fs;
    float blend = 1.0f - std::exp(-t / 0.7f);
    float gTarget[3];
    for (int s = 0; s < voice.numStrings; s++) {
      float gSus = voice.gFast[s] + (voice.gSlow[s] - voice.gFast[s]) * blend;
      gTarget[s] = is_sustained ? gSus : voice.gDamp[s];
    }

    for (int i = 0; i < AUDIO_BLOCK_SIZE; i++) {
      if (voice.fadeCount) {
        voice.fadeCount--;
        if (voice.fadeCount == 0) {
          voice.gain = 0.0f;
          if (voice.hasPending) {
            startVoice(voice, (int)voice.pendNote, voice.pendVel, voice.pendPedal);
          } else {
            voice.active = false;
          }
          break;
        }
        float u = (float)voice.fadeCount * (1.0f / (float)VOICE_RELEASE_SAMPLES);
        voice.gain = u * u * (3.0f - 2.0f * u);
      }

      // 读取各弦当前位移
      float r[3] = { 0.0f, 0.0f, 0.0f };
      int wIdx[3];
      for (int s = 0; s < voice.numStrings; s++) {
        float* buf = voice.delay[s];
        int w = voice.writeIdx[s];
        wIdx[s] = w;
        float rp = (float)w - voice.delayLen[s];
        if (rp < 0.0f) rp += (float)MAX_DELAY_LEN;
        int i0 = (int)rp;
        float frac = rp - (float)i0;
        int i1 = i0 + 1; if (i1 >= MAX_DELAY_LEN) i1 = 0;
        r[s] = buf[i0] + (buf[i1] - buf[i0]) * frac;
      }

      // 同音弦桥耦合
      float sum = 0.0f;
      for (int s = 0; s < voice.numStrings; s++) sum += r[s];
      float coupled[3];
      for (int s = 0; s < voice.numStrings; s++) {
        coupled[s] = r[s] * (1.0f - STRING_CROSS * (voice.numStrings - 1))
                   + (sum - r[s]) * STRING_CROSS;
      }

      float out = 0.0f;
      int k = voice.injectCount;
      for (int s = 0; s < voice.numStrings; s++) {
        float x = coupled[s];
        x = voice.loopLP[s].process(x);
        for (int m = 0; m < NUM_AP; m++) x = voice.disp[s][m].process(x);

        voice.feedback[s] += 0.004f * (gTarget[s] - voice.feedback[s]);
        float nxt = x * voice.feedback[s];

        // 锤击激励注入 (速度相关脉冲 + 毛毡噪声)
        if (k < voice.injectLen) {
          float u = (float)k / (float)(voice.injectLen - 1);
          float win = std::sin(PI_F * u);
          float pulse = win * win * (1.0f - 0.5f * u);
          float sgn = (s == 0) ? 1.0f : ((s == 1) ? 0.95f : 0.90f);
          nxt += pulse * voice.injectAmp * sgn;
          nxt += whiteNoise() * voice.noiseAmp * sgn * std::exp(-(float)k / voice.noiseDecay);
        }

        voice.delay[s][wIdx[s]] = nxt;
        int w = wIdx[s] + 1; if (w >= MAX_DELAY_LEN) w = 0;
        voice.writeIdx[s] = w;

        out += r[s];
      }
      voice.injectCount++;

      out *= 1.0f / (float)voice.numStrings;

      if (voice.damperCount > 0) {
        out += whiteNoise() * 0.02f * ((float)voice.damperCount / (0.02f * fs));
        voice.damperCount--;
      }

      float g2 = out * voice.gain * VOICE_MIX_SCALE;
      block_mix_L[i] += g2 * voice.panL;
      block_mix_R[i] += g2 * voice.panR;

      voice.ageSamples++;

      if (voice.fadeCount) continue;
      if (std::abs(out) < 1e-6f) {
        voice.silentCount++;
        if (voice.silentCount > 4000) { voice.active = false; voice.gain = 0.0f; }
      } else {
        voice.silentCount = 0;
      }
    }
  }

  // 主增益平滑
  float target_gain = (float)master_gain_q15 / 32767.0f;
  if (!gain_smooth_initialized) {
    master_gain_smooth = target_gain;
    gain_smooth_initialized = true;
  } else {
    master_gain_smooth += 0.005f * (target_gain - master_gain_smooth);
  }

  for (int i = 0; i < AUDIO_BLOCK_SIZE; i++) {
    float dl = block_mix_L[i];
    float dr = block_mix_R[i];

    // 声板共鸣
    float sbL = soundboard.processL(dl);
    float sbR = soundboard.processR(dr);
    float bodyL = dl * 0.70f + sbL * 0.30f;
    float bodyR = dr * 0.70f + sbR * 0.30f;

    // 添加空间感
    float wetL, wetR;
    reverb.process(bodyL, bodyR, wetL, wetR);
    float fl = bodyL + wetL;
    float fr = bodyR + wetR;

    fl *= master_gain_smooth;
    fr *= master_gain_smooth;

    // DC 阻断
    float outL = fl - dc_prev_in_L + DC_BLOCK_R * dc_prev_out_L;
    dc_prev_in_L = fl; dc_prev_out_L = outL;
    float outR = fr - dc_prev_in_R + DC_BLOCK_R * dc_prev_out_R;
    dc_prev_in_R = fr; dc_prev_out_R = outR;
    if (std::abs(dc_prev_in_L) < 1e-20f) dc_prev_in_L = 0.0f;
    if (std::abs(dc_prev_out_L) < 1e-20f) dc_prev_out_L = 0.0f;
    if (std::abs(dc_prev_in_R) < 1e-20f) dc_prev_in_R = 0.0f;
    if (std::abs(dc_prev_out_R) < 1e-20f) dc_prev_out_R = 0.0f;

    outL = softClip(outL);
    outR = softClip(outR);

    float scaledL = outL * 30000.0f;
    float scaledR = outR * 30000.0f;
    int32_t sL = (int32_t)((scaledL >= 0.0f) ? (scaledL + 0.5f) : (scaledL - 0.5f));
    int32_t sR = (int32_t)((scaledR >= 0.0f) ? (scaledR + 0.5f) : (scaledR - 0.5f));
    if (sL > 32767) sL = 32767; if (sL < -32768) sL = -32768;
    if (sR > 32767) sR = 32767; if (sR < -32768) sR = -32768;

    pcm_out[i * 2]     = (int16_t)sL;
    pcm_out[i * 2 + 1] = (int16_t)sR;
  }
}

// ---------------------------------------------------------------------------
//  写入标准 WAV 文件头
// ---------------------------------------------------------------------------
void writeWavHeader(std::ofstream& out, uint32_t totalSamples) {
  uint32_t subchunk2Size = totalSamples * 2 * sizeof(int16_t);
  uint32_t chunkSize = 36 + subchunk2Size;
  uint32_t sampleRate = SAMPLE_RATE;
  uint32_t byteRate = SAMPLE_RATE * 2 * sizeof(int16_t);
  uint16_t blockAlign = 2 * sizeof(int16_t);
  uint16_t bitsPerSample = 16;
  uint16_t numChannels = 2;
  uint16_t audioFormat = 1;

  out.write("RIFF", 4);
  out.write(reinterpret_cast<const char*>(&chunkSize), 4);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  uint32_t subchunk1Size = 16;
  out.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
  out.write(reinterpret_cast<const char*>(&audioFormat), 2);
  out.write(reinterpret_cast<const char*>(&numChannels), 2);
  out.write(reinterpret_cast<const char*>(&sampleRate), 4);
  out.write(reinterpret_cast<const char*>(&byteRate), 4);
  out.write(reinterpret_cast<const char*>(&blockAlign), 2);
  out.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&subchunk2Size), 4);
}

int main() {
  std::cout << "Starting Audio Physics Model Simulator (v" << SYNTH_VERSION << ")..." << std::endl;

  precomputeNoteTables();
  soundboard.init((float)SAMPLE_RATE);
  reverb.init();

  for (int v = 0; v < VOICE_COUNT; v++) {
    PianoVoice& pv = voices[v];
    pv.active = false;
    pv.age = 0;
    pv.pedal = 0;
    pv.silentCount = 0;
    pv.numStrings = 1;
    pv.gain = 0.0f;
    pv.fadeCount = 0;
    pv.hasPending = false;
    pv.damperCount = 0;
    pv.wasSustained = false;
    pv.panL = 0.5f;
    pv.panR = 0.5f;
    pv.ageSamples = 0;
    for (int s = 0; s < 3; s++) {
      pv.writeIdx[s] = 0;
      pv.delayLen[s] = 4.0f;
      pv.feedback[s] = 0.0f;
      pv.gFast[s] = 0.999f; pv.gSlow[s] = 0.999f; pv.gDamp[s] = 0.98f;
      pv.loopLP[s].a = 0.9f; pv.loopLP[s].reset();
      for (int m = 0; m < NUM_AP; m++) { pv.disp[s][m].a = AP_COEFF; pv.disp[s][m].reset(); }
    }
  }

  // 模拟播放《生日快乐》谱面测试
  const uint8_t birthday_notes[] = {
    24, 24, 26, 24, 29, 28,  24, 24, 26, 24, 31, 29,
    24, 24, 36, 33, 29, 28, 26,  34, 34, 33, 29, 31, 29
  };
  const uint16_t birthday_durations[] = {
    360, 120, 480, 480, 480, 960, 360, 120, 480, 480, 480, 960,
    360, 120, 480, 480, 480, 480, 960, 360, 120, 480, 480, 480, 1200
  };
  const uint8_t birthday_velocity[] = {
    85, 75, 110, 95, 115, 100, 85, 75, 110, 95, 120, 100,
    85, 75, 125, 110, 100, 95, 105, 110, 100, 115, 100, 110, 105
  };
  const uint8_t birthday_pedal[] = {
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0
  };
  size_t note_count = sizeof(birthday_notes) / sizeof(birthday_notes[0]);

  std::vector<int16_t> pcm_total;
  int16_t block_pcm[AUDIO_BLOCK_SIZE * 2];

  for (size_t n = 0; n < note_count; n++) {
    global_sustain_pedal = birthday_pedal[n];
    triggerPianoNote(birthday_notes[n], birthday_velocity[n], birthday_pedal[n]);

    uint32_t note_samples = (uint32_t)((float)birthday_durations[n] * SAMPLE_RATE / 1000.0f);
    uint32_t blocks = note_samples / AUDIO_BLOCK_SIZE;

    for (uint32_t b = 0; b < blocks; b++) {
      renderBlock(block_pcm);
      pcm_total.insert(pcm_total.end(), block_pcm, block_pcm + AUDIO_BLOCK_SIZE * 2);
    }
  }

  // 尾音: 让余韵与混响自然收尾
  uint32_t tail_blocks = (uint32_t)(2.0f * SAMPLE_RATE / AUDIO_BLOCK_SIZE);
  global_sustain_pedal = 1;
  for (uint32_t b = 0; b < tail_blocks; b++) {
    renderBlock(block_pcm);
    pcm_total.insert(pcm_total.end(), block_pcm, block_pcm + AUDIO_BLOCK_SIZE * 2);
  }

  std::ofstream wavFile("output.wav", std::ios::binary);
  if (wavFile.is_open()) {
    writeWavHeader(wavFile, pcm_total.size() / 2);
    wavFile.write(reinterpret_cast<const char*>(pcm_total.data()), pcm_total.size() * sizeof(int16_t));
    wavFile.close();
    std::cout << "Successfully generated 'output.wav'! Total samples: " << pcm_total.size() / 2 << std::endl;
  } else {
    std::cerr << "Failed to write WAV file!" << std::endl;
  }

  return 0;
}

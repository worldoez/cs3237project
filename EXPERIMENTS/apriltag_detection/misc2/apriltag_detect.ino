/*
 * TensorFlow Lite AprilTag Detection on ESP32-CAM
 * 
 * Hardware: ESP32-CAM (AI-Thinker or similar)
 * Requirements:
 * - ESP32 board support in Arduino IDE
 * - TensorFlow Lite Micro library
 * - apriltag_multitask_quant_tflite.h (your model)
 */

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Include your model
#include "apriltag_multitask_quant_tflite.h"

// Camera pin definitions for AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// LED flash (optional, for lighting)
#define LED_FLASH_GPIO     4

// TensorFlow Lite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  // Use PSRAM for tensor arena (ESP32-CAM has 4MB PSRAM)
  constexpr int kTensorArenaSize = 150000; // 150KB - adjust as needed
  uint8_t* tensor_arena = nullptr;
  
  tflite::AllOpsResolver resolver;
  
  // Expected input dimensions (adjust based on your model)
  int input_width = 0;
  int input_height = 0;
  int input_channels = 0;
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n\n=================================");
  Serial.println("ESP32-CAM AprilTag Detection");
  Serial.println("=================================\n");
  
  // Initialize LED flash (optional)
  pinMode(LED_FLASH_GPIO, OUTPUT);
  digitalWrite(LED_FLASH_GPIO, LOW);
  
  // Check if PSRAM is available
  if (psramFound()) {
    Serial.println("✓ PSRAM found");
    Serial.print("  Free PSRAM: ");
    Serial.print(ESP.getFreePsram() / 1024);
    Serial.println(" KB");
  } else {
    Serial.println("✗ PSRAM not found - may have memory issues!");
  }
  
  // Initialize camera
  if (!initCamera()) {
    Serial.println("✗ Camera initialization failed!");
    while (true) delay(1000);
  }
  
  // Initialize TensorFlow Lite
  if (!initTFLite()) {
    Serial.println("✗ TensorFlow Lite initialization failed!");
    while (true) delay(1000);
  }
  
  Serial.println("\n✓ Setup complete!");
  Serial.println("Starting inference loop...\n");
  
  delay(2000);
}

void loop() {
  // Capture image from camera
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("✗ Camera capture failed");
    delay(1000);
    return;
  }
  
  Serial.print("✓ Image captured: ");
  Serial.print(fb->width);
  Serial.print("x");
  Serial.print(fb->height);
  Serial.print(" (");
  Serial.print(fb->len);
  Serial.println(" bytes)");
  
  // Optional: Turn on LED flash for better lighting
  // digitalWrite(LED_FLASH_GPIO, HIGH);
  // delay(100);
  
  // Preprocess image and fill input tensor
  preprocessImage(fb);
  
  // Optional: Turn off LED flash
  // digitalWrite(LED_FLASH_GPIO, LOW);
  
  // Run inference
  unsigned long start_time = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long end_time = micros();
  
  // Return frame buffer to camera
  esp_camera_fb_return(fb);
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("✗ Inference failed!");
    delay(1000);
    return;
  }
  
  // Print inference time
  Serial.print("⏱ Inference time: ");
  Serial.print((end_time - start_time) / 1000.0);
  Serial.println(" ms");
  
  // Process and display results
  processAprilTagOutput();
  
  Serial.println("---\n");
  delay(2000); // Adjust delay based on your needs
}

bool initCamera() {
  Serial.println("Initializing camera...");
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE; // Use GRAYSCALE for AprilTags
  
  // Frame size - adjust based on your model's input size
  // Options: QQVGA(160x120), QCIF(176x144), HQVGA(240x176), QVGA(320x240)
  config.frame_size = FRAMESIZE_QVGA; // 320x240 - adjust to match your model
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  // Use PSRAM for frame buffer
  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
  }
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("✗ Camera init failed: 0x%x\n", err);
    return false;
  }
  
  // Get camera sensor and adjust settings
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0 = no effect
    s->set_whitebal(s, 1);       // white balance
    s->set_awb_gain(s, 1);       // auto white balance gain
    s->set_wb_mode(s, 0);        // white balance mode
    s->set_exposure_ctrl(s, 1);  // auto exposure
    s->set_aec2(s, 0);           // automatic exposure control
    s->set_gain_ctrl(s, 1);      // auto gain
    s->set_agc_gain(s, 0);       // gain value
    s->set_gainceiling(s, (gainceiling_t)0); // gain ceiling
    s->set_bpc(s, 0);            // black pixel correction
    s->set_wpc(s, 1);            // white pixel correction
    s->set_raw_gma(s, 1);        // gamma correction
    s->set_lenc(s, 1);           // lens correction
    s->set_hmirror(s, 0);        // horizontal mirror
    s->set_vflip(s, 0);          // vertical flip
  }
  
  Serial.println("✓ Camera initialized");
  return true;
}

bool initTFLite() {
  Serial.println("Initializing TensorFlow Lite...");
  
  // Allocate tensor arena in PSRAM if available
  if (psramFound()) {
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    if (tensor_arena == nullptr) {
      Serial.println("✗ Failed to allocate tensor arena in PSRAM");
      return false;
    }
    Serial.println("✓ Tensor arena allocated in PSRAM");
  } else {
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    if (tensor_arena == nullptr) {
      Serial.println("✗ Failed to allocate tensor arena");
      return false;
    }
    Serial.println("✓ Tensor arena allocated in DRAM");
  }
  
  // Load model
  model = tflite::GetModel(apriltag_multitask_quant_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("✗ Model version mismatch! Expected: %d, Got: %d\n",
                  TFLITE_SCHEMA_VERSION, model->version());
    return false;
  }
  Serial.println("✓ Model loaded");
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("✗ AllocateTensors() failed");
    return false;
  }
  Serial.println("✓ Tensors allocated");
  
  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Store input dimensions
  if (input->dims->size >= 3) {
    input_height = input->dims->data[1];
    input_width = input->dims->data[2];
    input_channels = (input->dims->size == 4) ? input->dims->data[3] : 1;
  }
  
  // Print model info
  Serial.println("\n📊 Model Information:");
  Serial.print("  Input shape: [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  Serial.print("  Input type: ");
  printTensorType(input->type);
  Serial.print("  Output shape: [");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  Serial.print("  Output type: ");
  printTensorType(output->type);
  Serial.print("  Arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");
  
  return true;
}

void preprocessImage(camera_fb_t* fb) {
  // This function needs to be customized based on your model's requirements
  // Common preprocessing steps:
  // 1. Resize image to match input dimensions
  // 2. Convert color format if needed
  // 3. Normalize pixel values
  // 4. Quantize if model expects uint8/int8
  
  if (input->type == kTfLiteUInt8) {
    uint8_t* input_data = input->data.uint8;
    
    // Simple nearest-neighbor resize (you may want bilinear for better quality)
    int img_width = fb->width;
    int img_height = fb->height;
    
    for (int y = 0; y < input_height; y++) {
      for (int x = 0; x < input_width; x++) {
        // Map output coordinates to input coordinates
        int src_x = (x * img_width) / input_width;
        int src_y = (y * img_height) / input_height;
        
        // Get pixel value (grayscale)
        uint8_t pixel = fb->buf[src_y * img_width + src_x];
        
        // Write to input tensor
        int idx = y * input_width + x;
        if (input_channels == 3) {
          // If model expects RGB, replicate grayscale to all channels
          input_data[idx * 3 + 0] = pixel; // R
          input_data[idx * 3 + 1] = pixel; // G
          input_data[idx * 3 + 2] = pixel; // B
        } else {
          // Single channel
          input_data[idx] = pixel;
        }
      }
    }
  } else if (input->type == kTfLiteInt8) {
    int8_t* input_data = input->data.int8;
    
    int img_width = fb->width;
    int img_height = fb->height;
    
    for (int y = 0; y < input_height; y++) {
      for (int x = 0; x < input_width; x++) {
        int src_x = (x * img_width) / input_width;
        int src_y = (y * img_height) / input_height;
        
        uint8_t pixel = fb->buf[src_y * img_width + src_x];
        
        // Convert uint8 [0,255] to int8 [-128,127]
        int8_t quantized = (int8_t)(pixel - 128);
        
        int idx = y * input_width + x;
        if (input_channels == 3) {
          input_data[idx * 3 + 0] = quantized;
          input_data[idx * 3 + 1] = quantized;
          input_data[idx * 3 + 2] = quantized;
        } else {
          input_data[idx] = quantized;
        }
      }
    }
  } else if (input->type == kTfLiteFloat32) {
    float* input_data = input->data.f;
    
    int img_width = fb->width;
    int img_height = fb->height;
    
    for (int y = 0; y < input_height; y++) {
      for (int x = 0; x < input_width; x++) {
        int src_x = (x * img_width) / input_width;
        int src_y = (y * img_height) / input_height;
        
        uint8_t pixel = fb->buf[src_y * img_width + src_x];
        
        // Normalize to [0, 1] or [-1, 1] based on your model
        float normalized = pixel / 255.0f;
        
        int idx = y * input_width + x;
        if (input_channels == 3) {
          input_data[idx * 3 + 0] = normalized;
          input_data[idx * 3 + 1] = normalized;
          input_data[idx * 3 + 2] = normalized;
        } else {
          input_data[idx] = normalized;
        }
      }
    }
  }
}

void processAprilTagOutput() {
  // This function processes your model's output
  // Customize based on what your multitask model returns
  // Common outputs: bounding boxes, tag IDs, corners, etc.
  
  Serial.println("🏷️  Detection Results:");
  
  if (output->type == kTfLiteUInt8) {
    uint8_t* output_data = output->data.uint8;
    int num_elements = output->bytes;
    
    // Example: Print first few output values
    int print_count = min(10, num_elements);
    for (int i = 0; i < print_count; i++) {
      Serial.printf("  Output[%d]: %d\n", i, output_data[i]);
    }
    
    // Add your custom logic here based on model output format
    // For example:
    // - Parse bounding boxes
    // - Extract tag IDs
    // - Get corner coordinates
    // - Calculate confidence scores
    
  } else if (output->type == kTfLiteFloat32) {
    float* output_data = output->data.f;
    int num_elements = output->bytes / sizeof(float);
    
    int print_count = min(10, num_elements);
    for (int i = 0; i < print_count; i++) {
      Serial.printf("  Output[%d]: %.4f\n", i, output_data[i]);
    }
  } else if (output->type == kTfLiteInt8) {
    int8_t* output_data = output->data.int8;
    int num_elements = output->bytes;
    
    int print_count = min(10, num_elements);
    for (int i = 0; i < print_count; i++) {
      Serial.printf("  Output[%d]: %d\n", i, output_data[i]);
    }
  }
  
  // Example: Check for detected tags (customize based on your model)
  // if (confidence > threshold) {
  //   Serial.printf("  ✓ AprilTag detected! ID: %d, Confidence: %.2f\n", tag_id, confidence);
  // } else {
  //   Serial.println("  ✗ No tags detected");
  // }
}

void printTensorType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: Serial.print("Float32"); break;
    case kTfLiteUInt8: Serial.print("UInt8"); break;
    case kTfLiteInt8: Serial.print("Int8"); break;
    case kTfLiteInt32: Serial.print("Int32"); break;
    case kTfLiteInt64: Serial.print("Int64"); break;
    default: Serial.printf("Unknown(%d)", type); break;
  }
  Serial.println();
}
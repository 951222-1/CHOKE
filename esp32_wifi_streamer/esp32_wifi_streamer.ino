#include "esp_camera.h"
#include <WiFi.h>
#include <driver/i2s.h>

// ==========================================
// 🌐 Wi-Fi 設定
// ==========================================
const char* ssid = "mkc_ssd";
const char* password = "1234567890@";

// ==========================================
// 📷 Seeed Studio XIAO ESP32S3 Sense 相機腳位定義
// ==========================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ==========================================
// 🎙️ XIAO ESP32S3 Sense 內建 PDM 麥克風腳位定義
// ==========================================
#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_42
#define I2S_MIC_SERIAL_DATA  GPIO_NUM_41

// ==========================================
// 🔌 串流伺服器 Port 設定
// ==========================================
WiFiServer videoServer(80);
WiFiServer audioServer(81);

// ==========================================
// 🛠️ 初始化相機
// ==========================================
bool initCamera() {
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
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA; // 320x240 解析度，串流較流暢
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  // 初始化相機
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("相機初始化失敗: 0x%x\n", err);
    return false;
  }
  return true;
}

// ==========================================
// 🛠️ 初始化 I2S PDM 麥克風
// ==========================================
bool initMicrophone() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
    .sample_rate = 16000, // 16kHz 取樣率
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT, // 16-bit 深度
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // 單聲道
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_PIN_NO_CHANGE,
    .ws_io_num = I2S_MIC_SERIAL_CLOCK,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_MIC_SERIAL_DATA
  };

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("I2S 驅動安裝失敗: 0x%x\n", err);
    return false;
  }

  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("I2S 設定接腳失敗: 0x%x\n", err);
    return false;
  }
  return true;
}

// ==========================================
// 🔄 FreeRTOS 影像串流任務
// ==========================================
void videoTask(void* pvParameters) {
  while (true) {
    WiFiClient client = videoServer.available();
    if (client) {
      Serial.println("🎥 新增影像連線");
      client.print("HTTP/1.1 200 OK\r\n");
      client.print("Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n");

      while (client.connected()) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
          Serial.println("無法取得相機畫面");
          delay(10);
          continue;
        }

        client.print("--frame\r\n");
        client.print("Content-Type: image/jpeg\r\n");
        client.print("Content-Length: " + String(fb->len) + "\r\n\r\n");
        client.write(fb->buf, fb->len);
        client.print("\r\n");

        esp_camera_fb_return(fb);
        delay(50); // 控制在 ~20 FPS 左右
      }
      client.stop();
      Serial.println("🎥 影像連線關閉");
    }
    delay(10);
  }
}

// ==========================================
// 🔄 FreeRTOS 聲音串流任務
// ==========================================
void audioTask(void* pvParameters) {
  int16_t mic_buffer[512];
  size_t bytes_read = 0;

  while (true) {
    WiFiClient client = audioServer.available();
    if (client) {
      Serial.println("🎙️ 新增聲音連線");
      client.print("HTTP/1.1 200 OK\r\n");
      client.print("Content-Type: audio/wav\r\n");
      client.print("Connection: close\r\n\r\n");

      while (client.connected()) {
        // 從 PDM 麥克風讀取音訊數據
        esp_err_t res = i2s_read(I2S_NUM_0, mic_buffer, sizeof(mic_buffer), &bytes_read, portMAX_DELAY);
        if (res == ESP_OK && bytes_read > 0) {
          client.write((const uint8_t*)mic_buffer, bytes_read);
        }
      }
      client.stop();
      Serial.println("🎙️ 聲音連線關閉");
    }
    delay(10);
  }
}

// ==========================================
// 主程式 Setup
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("⚡ 啟動 Seeed Studio XIAO ESP32S3 Sense...");

  // 初始化硬體
  if (!initCamera()) {
    Serial.println("❌ 相機初始化失敗");
  } else {
    Serial.println("✅ 相機初始化成功");
  }

  if (!initMicrophone()) {
    Serial.println("❌ PDM 麥克風初始化失敗");
  } else {
    Serial.println("✅ PDM 麥克風初始化成功");
  }

  // 連線 Wi-Fi
  Serial.printf("正在連線至 Wi-Fi: %s ", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Wi-Fi 連線成功");
  Serial.print("📡 ESP32-S3 IP 位址: ");
  Serial.println(WiFi.localIP());

  // 啟動 TCP 串流伺服器
  videoServer.begin();
  audioServer.begin();

  // 建立 FreeRTOS 多工任務（影像和聲音互不干擾）
  xTaskCreatePinnedToCore(videoTask, "videoTask", 4096, NULL, 5, NULL, 0);
  xTaskCreatePinnedToCore(audioTask, "audioTask", 4096, NULL, 5, NULL, 1);
}

void loop() {
  // FreeRTOS 任務在背景執行，loop() 保持空置
  delay(1000);
}

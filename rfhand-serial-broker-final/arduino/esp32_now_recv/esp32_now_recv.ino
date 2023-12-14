/*
  ESP-NOW Remote Sensor - Receiver
  esp-now-rcv.ino
  Receives Temperature & Humidity data from other ESP32 via ESP-NOW
  
  DroneBot Workshop 2022
  https://dronebotworkshop.com
*/

// Include required libraries
#include <WiFi.h>
#include <esp_now.h>

// Define data structure
#define row 24
#define col 48
typedef struct {
  uint16_t data[row][col];
} result_mat_t;

// Create structured data object
result_mat_t result;

// Callback function
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) 
{
  // Get incoming data
  memcpy(&result, incomingData, sizeof(result));
  
  // Print to Serial Monitor
  Serial.write((uint8_t *)result.data, sizeof(result.data));
}
 
void setup() {
  // Set up Serial Monitor
  Serial.begin(115200);

  // Start ESP32 in Station mode
  WiFi.mode(WIFI_STA);

  // Print ESP32's MAC Address
  Serial.print("MAC Address:");
  Serial.println(WiFi.macAddress());

  // Initalize ESP-NOW
  if (esp_now_init() != 0) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
   
  // Register callback function
  esp_now_register_recv_cb(OnDataRecv);
  Serial.println("registered callback");
}

void loop() {
  }
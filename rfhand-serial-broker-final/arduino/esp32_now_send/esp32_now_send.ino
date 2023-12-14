/*
  ESP-NOW Remote Sensor - Transmitter
  esp-now-xmit.ino
  Sends Temperature & Humidity data to other ESP32 via ESP-NOW
  Uses DHT22

  DroneBot Workshop 2022
  https://dronebotworkshop.com
*/

// Include required libraries
#include <WiFi.h>
#include <esp_now.h>

// Variables for temperature and humidity
float temp;
float humid;

// Responder MAC Address (Replace with your responders MAC Address)
uint8_t broadcastAddress[] = {0xC8, 0xF0, 0x9E, 0x2B, 0xFC, 0x1C};

// Define data structure
#define row 24
#define col 48
typedef struct {
  uint16_t data[row][col];
} result_mat_t;

// Create structured data object
result_mat_t result;

// Register peer
esp_now_peer_info_t peerInfo;

// Sent data callback function
void OnDataSent(const uint8_t *macAddr, esp_now_send_status_t status)
{
  Serial.print("Last Packet Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

void setup() {

  // Setup Serial monitor
  Serial.begin(115200);
  delay(100);

  // Set ESP32 WiFi mode to Station temporarly
  WiFi.mode(WIFI_STA);

  // Initialize ESP-NOW
  if (esp_now_init() != 0) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Define callback
  esp_now_register_send_cb(OnDataSent);

  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  for (unsigned char i = 0; i < row; i++) {
    for (unsigned char j = 0; j < col; j++) {
      result.data[i][j] = i * 0xFF + j;
    }
  }

}

void loop() {
  // Add to structured data object
  // Send data
  esp_now_send(broadcastAddress, (uint8_t *) &result.data, 128);

}
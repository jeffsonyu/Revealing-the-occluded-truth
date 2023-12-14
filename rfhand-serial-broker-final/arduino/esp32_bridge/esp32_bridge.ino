extern "C" {
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
}

#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <WiFiServer.h>

#define CONFIG_WIFI_SSID "mano"
#define CONFIG_WIFI_PASSWORD "00000000"

#define CONFIG_WLAN_LOCAL_IP IPAddress(192, 168, 5, 1)
#define CONFIG_WLAN_GATEWAY IPAddress(192, 168, 5, 1)
#define CONFIG_WLAN_SUBNETMASK IPAddress(255, 255, 255, 0)

#define CONFIG_SERVER_TCP_PORT 8080

WiFiServer server(CONFIG_SERVER_TCP_PORT);
uint8_t wifi_read_buffer[1024] = { 0 };

void setup() {
  Serial.begin(921600);
  delay(500);
  // setCpuFrequencyMhz(240);

  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(CONFIG_WLAN_LOCAL_IP, CONFIG_WLAN_GATEWAY, CONFIG_WLAN_SUBNETMASK);
  WiFi.softAP(CONFIG_WIFI_SSID, CONFIG_WIFI_PASSWORD);
  esp_wifi_set_max_tx_power(78);  //maximum is 78
  esp_wifi_set_ps(WIFI_PS_NONE);

  server.begin();
}

// CircularBuffer<uint8_t, 4096> wifi_ring_buffer;
// void readTask(void* parameter) {
//   while (1) {
//     WiFiClient client = server.available();

//     if (client) {
//       while (client.connected()) {
//         if (client.available()) {
//           uint8_t buff[1024];  // Buffer of 1KB
//           int bytesReceived = client.readBytes(buff, sizeof(buff));

//           if (bytesReceived > 0) {
//             for (int i = 0; i < bytesReceived; i++) {
//               wifi_ring_buffer.push(buff[i]);
//             }
//           }
//         }
//       }
//       client.stop();
//     }
//   }

//   vTaskDelete(NULL);
// }

// void writeTask(void* parameter) {
//   while (1) {
//     while (!wifi_ring_buffer.isEmpty()) {
//       uint8_t data = wifi_ring_buffer.shift();
//       Serial.write(data);
//     }

//     vTaskDelay(10 / portTICK_PERIOD_MS);
//   }

//   vTaskDelete(NULL);
// }


void loop() {
  WiFiClient client = server.available();

  if (client) {
    while (client.connected()) {
      if (client.available()) {
        int len = client.readBytes(wifi_read_buffer, sizeof(wifi_read_buffer));
        Serial.write(wifi_read_buffer, len);
      }
    }

    client.stop();
  }
}
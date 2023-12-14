#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>

// WiFi network details
const char* ssid = "mano";
const char* password = "00000000";

// WiFi static ip, subnet, gateway
IPAddress staticIP(192, 168, 5, 2);
IPAddress subnet(255, 255, 255, 0);
IPAddress gateway(192, 168, 5, 1);

// TCP server
WiFiServer server(8080);

// Array declaration
uint16_t result[24][48] = {0};

void setup() {
  // Initialize Serial
  Serial.begin(115200);
  
  // Connect to WiFi
  Serial.println("Connecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  // Wait until connected
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting...");
  }

  // Configure static IP
  if (!WiFi.config(staticIP, gateway, subnet)) {
    Serial.println("Failed to configure static IP");
  } else {
    Serial.println("IP successfully configured");
  }

  // Start the TCP server
  server.begin();

  // Print connection details
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Check if a client connected
  WiFiClient client = server.available();

  if (client) {
    // Wait until the client sends data
    while(!client.available()){
      delay(1);
    }

    // Client is connected, send the array
    Serial.println("Client connected");

    for(int i=0; i<24; i++){
      for(int j=0; j<48; j++){
        // Send the array value
        client.println(result[i][j]);
      }
    }
    
    // Close the connection
    client.stop();
    Serial.println("Client disconnected");
  }
}
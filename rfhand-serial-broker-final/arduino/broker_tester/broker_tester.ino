#include <Arduino.h>

#define HAND_HEIGHT 24
#define HAND_WIDTH 48

typedef struct {
  uint16_t tx_flag;
  uint16_t data_length;
  uint16_t total_hand_data[HAND_HEIGHT][HAND_WIDTH];
  uint16_t checksum;
} hand_packet_t;

static hand_packet_t g_packet;

void fill_random_hand_data(uint16_t* arr, int height, int width, uint16_t min, uint16_t max) {
  for (int i = 0; i < width * height; i++) {
    arr[i] = rand() % (max - min) + min;
  }
}

void prepare_packet(hand_packet_t* pkt) {
  pkt->tx_flag = 0xFFFF;
  pkt->data_length = sizeof(pkt->total_hand_data);
  fill_random_hand_data((uint16_t*)pkt->total_hand_data, HAND_HEIGHT, HAND_WIDTH, 0, (0x1 << 12));
  pkt->checksum = 0x0000 ^ 0xFFFF ^ pkt->data_length;

  for (int i = 0; i < HAND_HEIGHT; i++) {
    for (int j = 0; j < HAND_WIDTH; j++) {
      pkt->checksum ^= pkt->total_hand_data[i][j];
    }
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  srand(millis());
  prepare_packet(&g_packet);
  uint8_t * g_packet_buffer = (uint8_t *)&g_packet;
  Serial.write(g_packet_buffer, sizeof(g_packet));
  // put your main code here, to run repeatedly:
}
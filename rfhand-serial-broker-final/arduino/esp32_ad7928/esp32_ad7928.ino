/*20230316
  Tactile glove scanning circuit (24 row * 48 col)
  by Chunpeng Jiang
*/

#include <SPI.h>
#include <Arduino.h>
#include <WiFi.h>

// WiFi 配置
const char* wifi_ssid = "mano";
const char* wifi_password = "00000000";
// WiFi static ip, subnet, gateway
IPAddress staticIP(192, 168, 5, 2);
IPAddress subnet(255, 255, 255, 0);
IPAddress gateway(192, 168, 5, 1);

#define row 24
#define col 48

// AD7928的控制寄存器（12bit）：
#define ADC_COMMOND_SEQ 0xDF90         // 连续读取   SEQ=1 SHADOW=1;  1101 1111 1001 0000  （最后4位没有作用）
#define ADC_COMMOND_SINGLE_CH0 0x8310  // 单通道模式
#define ADC_COMMOND_SINGLE_CH1 0x8710
#define ADC_COMMOND_SINGLE_CH2 0x8B10
#define ADC_COMMOND_SINGLE_CH3 0x8F10
#define ADC_COMMOND_SINGLE_CH4 0x9310
#define ADC_COMMOND_SINGLE_CH5 0x9710
#define ADC_COMMOND_SINGLE_CH6 0x9B10
#define ADC_COMMOND_SINGLE_CH7 0x9F10

// SPI 接口配置
#define VSPI_ENABLED

#ifdef VSPI_ENABLED
boolean vspiEnabled = 1;
#else
boolean vspiEnabled = 0;
#endif


// VSPI SET PIN
#define VSPI_MOSI 23
#define VSPI_MISO 19
#define VSPI_SCLK 18
#define VSPI_SS0 5
#define VSPI_SS1 10
#define VSPI_SS2 9
#define VSPI_SS3 13
#define VSPI_SS4 14
#define VSPI_SS5 15

// SPI Define
static const int spiClk = 1000000;  // 1 MHz
SPIClass* vspi = NULL;
#define VSPI_FLAG 0

// 译码器定义
#define A0 25
#define A1 26
#define A2 27
#define A3 4

//译码器的管脚顺序：A0,A1,A2,A3
int DecoderPins[] = { 25, 26, 27, 4 };

//row0-23的value，参见《对应关系》表格，共24行
const unsigned char DecoderValue[24] = { 0x00, 0x01, 0x02, 0x03,
                                         0x04, 0x05, 0x06, 0x07,
                                         0x08, 0x09, 0x0A, 0x0B,
                                         0x0C, 0x0D, 0x0E, 0x0F,
                                         0x00, 0x01, 0x02, 0x03,
                                         0x04, 0x05, 0x06, 0x07 };


// 结果矩阵定义Result mat define
// uint16_t total_result_mat[row][col] = { 0 };
typedef struct {
  uint16_t data[row][col];
} result_mat_t;

result_mat_t total_result_mat = { 0 };


//串口发送定义serial port parameters
uint16_t tx_flag = 0xFFFF;
uint16_t data_length = sizeof(total_result_mat.data);  //24*48=1152=0x0480
uint8_t data_length_1st_byte = sizeof(total_result_mat.data) & 0xff;
uint8_t data_length_2nd_byte = (sizeof(total_result_mat.data) >> 8) & 0xff;
uint8_t data_header[] = { 0xFF, 0xFF, data_length_1st_byte, data_length_2nd_byte };


// 功能定义
#define __DEBUG_SERIAL_TXDATA__ 0
#define __CONFIG_USE_WIFI__ 0

// SPI 初始化
void spi_init() {

  if (vspiEnabled) {
    /////Serial.println("VSPI initializing...");
    vspi = new SPIClass(VSPI);

    vspi->begin();
    vspi->setClockDivider(SPI_CLOCK_DIV4);  // SPI_CLOCK_DIV4: 4MHz, SPI_CLOCK_DIV16: 1MHz
    vspi->setDataMode(SPI_MODE0);
    vspi->setBitOrder(MSBFIRST);
    pinMode(VSPI_SS0, OUTPUT);
    pinMode(VSPI_SS1, OUTPUT);
    pinMode(VSPI_SS2, OUTPUT);
    pinMode(VSPI_SS3, OUTPUT);
    pinMode(VSPI_SS4, OUTPUT);
    pinMode(VSPI_SS5, OUTPUT);
    /////Serial.println("VSPI initialize success.");
    delay(10);
  }
}


// 写数据到SPI的控制寄存器
void VSPI_SetReg(unsigned long registerValue) {
  unsigned char bytesNumber;
  unsigned char txBuffer[2] = { 0, 0 };
  txBuffer[1] = (registerValue >> 8) & 0x000000FF;  //取出寄存器的高字节，存在txBuffer[1]里（因为SPI一次只能传1个字节）
  txBuffer[0] = (registerValue >> 0) & 0x000000FF;  //取出寄存器的低字节，存在txBuffer[0]里

  if (vspiEnabled) {
    //VSPI的ADC0
    bytesNumber = 2;
    digitalWrite(VSPI_SS0, LOW);
    while (bytesNumber > 0) {
      vspi->transfer(txBuffer[bytesNumber - 1]);
      bytesNumber--;
    }
    digitalWrite(VSPI_SS0, HIGH);
    //Serial.print(" VSPI0 SetReg: ");
    //Serial.println(registerValue, HEX);

    //VSPI的ADC1
    bytesNumber = 2;
    digitalWrite(VSPI_SS1, LOW);
    while (bytesNumber > 0) {
      vspi->transfer(txBuffer[bytesNumber - 1]);
      bytesNumber--;
    }
    digitalWrite(VSPI_SS1, HIGH);
    //Serial.print(" VSPI1 SetReg: ");
    //Serial.println(registerValue, HEX);

    //VSPI的ADC2
    bytesNumber = 2;
    digitalWrite(VSPI_SS2, LOW);
    while (bytesNumber > 0) {
      vspi->transfer(txBuffer[bytesNumber - 1]);
      bytesNumber--;
    }
    digitalWrite(VSPI_SS2, HIGH);
    //Serial.print(" VSPI2 SetReg: ");
    //Serial.println(registerValue, HEX);

    //VSPI的ADC3
    bytesNumber = 2;
    digitalWrite(VSPI_SS3, LOW);
    while (bytesNumber > 0) {
      vspi->transfer(txBuffer[bytesNumber - 1]);
      bytesNumber--;
    }
    digitalWrite(VSPI_SS3, HIGH);
    //Serial.print(" VSPI3 SetReg: ");
    //Serial.println(registerValue, HEX);

    //VSPI的ADC4
    bytesNumber = 2;
    digitalWrite(VSPI_SS4, LOW);
    while (bytesNumber > 0) {
      vspi->transfer(txBuffer[bytesNumber - 1]);
      bytesNumber--;
    }
    digitalWrite(VSPI_SS4, HIGH);
    //Serial.print(" VSPI4 SetReg: ");
    //Serial.println(registerValue, HEX);

    //VSPI的ADC5
    bytesNumber = 2;
    digitalWrite(VSPI_SS5, LOW);
    while (bytesNumber > 0) {
      vspi->transfer(txBuffer[bytesNumber - 1]);
      bytesNumber--;
    }
    digitalWrite(VSPI_SS5, HIGH);
    //Serial.print(" VSPI5 SetReg: ");
    //Serial.println(registerValue, HEX);
  }
}


// 读ADC
//方式1： 读取单个ADC的单通道值
uint16_t ReadSPIADCxCHx(unsigned char ADCSEL, unsigned long registerValue) {
  unsigned char bytesNumber = 2;  //transfer byte numbers
  byte inByte = 0;                // incoming byte from the SPI
  uint16_t result = 0;            // result to return
  unsigned char txBuffer[2] = { 0, 0 };
  txBuffer[1] = (registerValue >> 8) & 0x000000FF;
  txBuffer[0] = (registerValue >> 0) & 0x000000FF;

  // write register value to VSPI ADCx
  digitalWrite(ADCSEL, LOW);
  while (bytesNumber > 0) {
    vspi->transfer(txBuffer[bytesNumber - 1]);
    bytesNumber--;
  }
  digitalWrite(ADCSEL, HIGH);
  delay(10);

  // Read HSPI ADC two bytes: 1bit_0 + 3bit_chs + 12bit_data
  bytesNumber = 2;
  digitalWrite(ADCSEL, LOW);
  while (bytesNumber > 0) {
    result = result << 8;
    inByte = vspi->transfer(0x00);
    result = result | inByte;
    bytesNumber--;
  }
  digitalWrite(ADCSEL, HIGH);
  /////Serial.print(" VSPI ADC sampled successfull\n");
  /////Serial.print("SET REG: ");
  /////Serial.print(txBuffer[1], HEX);
  /////Serial.print(txBuffer[0], HEX);
  /////Serial.print("\n");
  /////Serial.println(result, HEX);
  delay(10);
  return (result);
}


//方式2： 顺序读取ADC0-5的CH0-7的值
void ReadSEQVSPI(unsigned long registerValue, int rowNum, result_mat_t* res) {
  //uint16_t result_mat[row] = {0};   // result matrix to return
  byte inByte = 0;      // incoming byte from the SPI
  uint16_t result = 0;  // result for single channel
  //setregistervalue
  unsigned char txBuffer[2] = { 0, 0 };
  unsigned char channel = 0;  //sampled ADC channel
  unsigned char bytesNumber = 2;
  txBuffer[1] = (registerValue >> 8) & 0x000000FF;
  txBuffer[0] = (registerValue >> 0) & 0x000000FF;

  // VSPI_SetReg(registerValue); // set initial ADC reg

  // read VSPI ADC0 8 Channels
  for (channel = 0; channel < 8; channel++) {
    if (channel < 7)
      txBuffer[1] = (txBuffer[1] & 0x000000E3) | ((channel + 1) << 2);
    else
      txBuffer[1] = (txBuffer[1] & 0x000000E3);
    bytesNumber = 2;
    result = 0;
    digitalWrite(VSPI_SS0, LOW);
    while (bytesNumber > 0) {
      result = result << 8;
      inByte = vspi->transfer(txBuffer[bytesNumber - 1]);
      result = result | inByte;
      bytesNumber--;
    }
    digitalWrite(VSPI_SS0, HIGH);
    res->data[rowNum][channel] = result & 0x0FFF;  //存在矩阵的前8个
  }
  //delay(10);

  // read VSPI ADC1 8 Channels
  for (channel = 0; channel < 8; channel++) {
    if (channel < 7)
      txBuffer[1] = (txBuffer[1] & 0x000000E3) | ((channel + 1) << 2);
    else
      txBuffer[1] = (txBuffer[1] & 0x000000E3);
    bytesNumber = 2;
    result = 0;
    digitalWrite(VSPI_SS1, LOW);
    while (bytesNumber > 0) {
      result = result << 8;
      inByte = vspi->transfer(txBuffer[bytesNumber - 1]);
      result = result | inByte;
      bytesNumber--;
    }
    digitalWrite(VSPI_SS1, HIGH);
    res->data[rowNum][8 + channel] = result & 0x0FFF;  //存在矩阵的前8个
  }
  //delay(10);

  // read VSPI ADC2 8 Channels
  for (channel = 0; channel < 8; channel++) {
    if (channel < 7)
      txBuffer[1] = (txBuffer[1] & 0x000000E3) | ((channel + 1) << 2);
    else
      txBuffer[1] = (txBuffer[1] & 0x000000E3);
    bytesNumber = 2;
    result = 0;
    digitalWrite(VSPI_SS2, LOW);
    while (bytesNumber > 0) {
      result = result << 8;
      inByte = vspi->transfer(txBuffer[bytesNumber - 1]);
      result = result | inByte;
      bytesNumber--;
    }
    digitalWrite(VSPI_SS2, HIGH);
    res->data[rowNum][16 + channel] = result & 0x0FFF;  //存在矩阵的前8个
  }
  // delay(10);

  // read VSPI ADC3 8 Channels
  for (channel = 0; channel < 8; channel++) {
    if (channel < 7)
      txBuffer[1] = (txBuffer[1] & 0x000000E3) | ((channel + 1) << 2);
    else
      txBuffer[1] = (txBuffer[1] & 0x000000E3);
    bytesNumber = 2;
    result = 0;
    digitalWrite(VSPI_SS3, LOW);
    while (bytesNumber > 0) {
      result = result << 8;
      inByte = vspi->transfer(txBuffer[bytesNumber - 1]);
      result = result | inByte;
      bytesNumber--;
    }
    digitalWrite(VSPI_SS3, HIGH);
    res->data[rowNum][24 + channel] = result & 0x0FFF;  //存在矩阵的前8个
  }
  // delay(10);

  // read VSPI ADC4 8 Channels
  for (channel = 0; channel < 8; channel++) {
    if (channel < 7)
      txBuffer[1] = (txBuffer[1] & 0x000000E3) | ((channel + 1) << 2);
    else
      txBuffer[1] = (txBuffer[1] & 0x000000E3);
    bytesNumber = 2;
    result = 0;
    digitalWrite(VSPI_SS4, LOW);
    while (bytesNumber > 0) {
      result = result << 8;
      inByte = vspi->transfer(txBuffer[bytesNumber - 1]);
      result = result | inByte;
      bytesNumber--;
    }
    digitalWrite(VSPI_SS4, HIGH);
    res->data[rowNum][32 + channel] = result & 0x0FFF;  //存在矩阵的前8个的后面8个
  }
  // delay(10);

  // read VSPI ADC5 8 Channels
  for (channel = 0; channel < 8; channel++) {
    if (channel < 7)
      txBuffer[1] = (txBuffer[1] & 0x000000E3) | ((channel + 1) << 2);
    else
      txBuffer[1] = (txBuffer[1] & 0x000000E3);
    bytesNumber = 2;
    result = 0;
    digitalWrite(VSPI_SS5, LOW);
    while (bytesNumber > 0) {
      result = result << 8;
      inByte = vspi->transfer(txBuffer[bytesNumber - 1]);
      result = result | inByte;
      bytesNumber--;
    }
    digitalWrite(VSPI_SS5, HIGH);
    res->data[rowNum][40 + channel] = result & 0x0FFF;  //存在矩阵前8个的后面8个的后面8个
  }
  // delay(10);
}


//2.4 初始化ADC的函数
void ADCReset() {
  /////Serial.println("\nReset AD7928...");
  char i = 0;
  for (i = 0; i < 6; i++)  //DUMMY PERIOD 上电后执行两个伪周期
  {
    //reset VSPI0_ADCXXX
    VSPI_SetReg(0xFFFF);
    delay(5);
  }
  VSPI_SetReg(ADC_COMMOND_SINGLE_CH0);
  delay(5);
}


//2.5 电压转换
float DataToVoltage(uint16_t rawData) {
  float voltage = 0;
  //char mGain = 1;
  float mVref = 2.51;                                          //根据U2实测
  voltage = (float(rawData & 0x0FFF) / 4095.000) * mVref * 2;  //在MODE0下，rawData不要右移（针对V4 PCB的情况）
  return (voltage);
}


//2.6 译码器管脚赋值函数
uint16_t Pin_mat[4] = { 0 };  //暂存Pin脚值的矩阵

void WriteDecoderPins(unsigned char value) {
  for (int k = 0; k < 4; k++) {
    Pin_mat[k] = bitRead(value, k);
    digitalWrite(DecoderPins[k], bitRead(value, k));
  }
}


//串口数据发送函数
void serial_txdata(result_mat_t* res) {

  uint16_t check_flag = 0x0000;  //two bytes of check flag

  // 异或生成校验字节
  check_flag = (check_flag ^ tx_flag) ^ data_length;
  for (unsigned char i = 0; i < row; i++) {
    for (unsigned char j = 0; j < col; j++) {
      check_flag = res->data[i][j] ^ check_flag;
    }
  }


  Serial.write(data_header, sizeof(data_header));

  // transfer datas
  for (unsigned char i = 0; i < row; i++) {
    for (unsigned char j = 0; j < col; j++) {
      // Serial.write(uint8_t((res->data[i][j] >> 8) & 0xff));
      // Serial.write(uint8_t((res->data[i][j]) & 0xff));
      Serial.write((uint8_t*)&(res->data[i][j]), 2);
    }
  }

  // transfer check flag
  Serial.write(uint8_t(check_flag & 0xff));
  Serial.write(uint8_t((check_flag >> 8) & 0xff));
}

#if __CONFIG_USE_WIFI__
WiFiClient client;

void wifi_txdata(result_mat_t* res) {

  uint16_t check_flag = 0x0000;  //two bytes of check flag

  // 异或生成校验字节
  check_flag = (check_flag ^ tx_flag) ^ data_length;
  for (unsigned char i = 0; i < row; i++) {
    for (unsigned char j = 0; j < col; j++) {
      check_flag = res->data[i][j] ^ check_flag;
    }
  }


  client.write(data_header, sizeof(data_header));

  // transfer datas
  for (unsigned char i = 0; i < row; i++) {
    for (unsigned char j = 0; j < col; j++) {
      // Serial.write(uint8_t((res->data[i][j] >> 8) & 0xff));
      // Serial.write(uint8_t((res->data[i][j]) & 0xff));
      client.write((uint8_t*)&(res->data[i][j]), 2);
    }
  }

  // transfer check flag
  client.write(uint8_t(check_flag & 0xff));
  client.write(uint8_t((check_flag >> 8) & 0xff));
  Serial.println("Send Success");
  if (WiFi.status() != WL_CONNECTED || !client.connected()) {
    ESP.restart();
  }
}
#endif




//初始化
void setup() {
  Serial.begin(921600);
  delay(5);
  while (!Serial) {
    ;
  }
  /////Serial.println("\n Serial to PC SETUP SUCCESSFUL");

  //译码器管脚模式设置
  pinMode(21, OUTPUT);
  pinMode(22, OUTPUT);
  pinMode(A0, OUTPUT);
  pinMode(A1, OUTPUT);
  pinMode(A2, OUTPUT);
  pinMode(A3, OUTPUT);

  //SPI初始化
  spi_init();

  //ADC重置
  ADCReset();

  Serial.println("Initializing Wi-Fi");

#if __CONFIG_USE_WIFI__
  // Connect to WiFi
  Serial.println("Connecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(wifi_ssid, wifi_password);

  // Wait until connected, timeout for 60s
  int timeout = 60;
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(WiFi.status());
    Serial.println("Connecting...");
    timeout--;
    if (timeout <= 0) {
      ESP.restart();
    }
  }

  // Configure static IP
  if (!WiFi.config(staticIP, gateway, subnet)) {
    Serial.println("Failed to configure static IP");
  } else {
    Serial.println("IP successfully configured");
  }

  // Print connection details
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
#endif
}

//---------------------------------------------------------------------------
void loop() {
#if __CONFIG_USE_WIFI__
  if (client.connect(gateway, 8080)) {
    Serial.println("Connected to server");
  } else {
    Serial.println("Failed to connect to server");
  }
#endif

#if __DEBUG_SERIAL_TXDATA__
  for (unsigned char i = 0; i < row; i++) {
    for (unsigned char j = 0; j < col; j++) {
      total_result_mat.data[i][j] = i * 0xFF + j;
    }
  }
  uint16_t offset = 0;

  while (1) {
    offset++;
    // for (unsigned char i = 0; i < row; i++) {
    //   for (unsigned char j = 0; j < col; j++) {
    //     Serial.printf("%d ", total_result_mat.data[i][j]);
    //   }
    //   Serial.println("");
    // }
    // Serial.println("-----------------");
#if __CONFIG_USE_WIFI__
    wifi_txdata(&total_result_mat);
#else
    serial_txdata(&total_result_mat);
    delay(50);
#endif
  }
#endif


  while (1) {
    // TODO: use multi-threading
    for (int i = 0; i < row; i++) {  //行扫描
      if (i < 16) {
        //第一块译码器工作（row0-15)，第二块不工作
        digitalWrite(21, LOW);   //1E1
        digitalWrite(22, HIGH);  //2E1
        WriteDecoderPins(DecoderValue[i]);
        //某一行拉低后，进行ADC0-5的CH0-7的循环读取
        ReadSEQVSPI(ADC_COMMOND_SINGLE_CH0, i, &total_result_mat);  //从CH0开始循环读取
        delay(1);
      } else {
        //第一块译码器不工作，第二块工作（row16-23)
        digitalWrite(21, HIGH);  //1E1
        digitalWrite(22, LOW);   //2E1
        WriteDecoderPins(DecoderValue[i]);
        //某一行拉低后，进行ADC0-5的CH0-7的循环读取
        ReadSEQVSPI(ADC_COMMOND_SINGLE_CH0, i, &total_result_mat);  //从CH0开始循环读取
        delay(1);
      }
    }

    //发送total_result_mat到上位机
#if __CONFIG_USE_WIFI__
    wifi_txdata(&total_result_mat);
#else
    serial_txdata(&total_result_mat);
#endif
    delay(1);
  }
}

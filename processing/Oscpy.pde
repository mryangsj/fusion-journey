import netP5.*;
import oscP5.*;
import ddf.minim.*;

int[][] coordinates;
int maxCoordinates = 100; // Maximum number of coordinates to process
int coordIndex = 0;
int cellCount = 70;
int[][] toneCell = {  { -14, -7, 0, 7, 14, 21},
                      { -13, -6, 1, 8, 15, 22},
                      { -12, -5, 2, 9, 16, 23},
                      { -11, -4, 3 , 10, 17 ,24},
                      { -10, -3, 4, 11, 18, 25},
                      { -9, -2, 5, 12, 19, 26}   };
int[] amp;
int amplIndex;
int toneIndex;

Minim minim;
AudioInput in;
OscP5 osc;
NetAddress supercollider;

void setup() {
  size(512, 512, P3D);
  coordinates = new int[maxCoordinates][2];

  // Initialize OSC
  osc = new OscP5(this, 12000);
  supercollider = new NetAddress("127.0.0.1", 57120);
  amp = new int[6];  
  frameRate(24);
  colorMode(HSB);
  minim = new Minim(this);
  in = minim.getLineIn(Minim.STEREO, 512);
}

void draw() {
  background(0);

  // Initialize the amplitude table
  for (int i = 0; i < amp.length; i++) {
    amp[i] = 0;
  }

  // Draw coordinates
  for (int i = 0; i < 10; i++) {
    int x = coordinates[i][0];
    int y = coordinates[i][1];
    drawCell(x, y);
  }

  // Draw the tone matrix
  drawToneMatrix();

  // Send OSC messages
  sendOSCmessages();
}

void oscEvent(OscMessage theOscMessage) {
  if (theOscMessage.checkAddrPattern("/coordinates")) {
    int x = theOscMessage.get(0).intValue();
    int y = theOscMessage.get(1).intValue();
    if (coordIndex < maxCoordinates) {
      coordinates[coordIndex][0] = x;
      coordinates[coordIndex][1] = y;
      coordIndex++;
    } else {
      coordIndex = 0; // Overwrite from the beginning if the number of coordinates exceeds the limit
    }
  }
}

void drawCell(int x, int y) {
  float size = random(5,25);
  fill(#00D20B,222);
  noStroke();
  ellipse(x, y, size, size);

  // Calculate cells captured by the point
  int cellWidth = width / 6;
  int cellHeight = height / 6;
  int gridX = int(map(x, 0, width, 0, 6));
  int gridY = int(map(y, 0, height, 0, 6));
  int radiusCellsX = int(map(size / 2, 0, width / 2, 0, 3));
  int radiusCellsY = int(map(size / 2, 0, height / 2, 0, 3));

  for (int i = max(0, gridX - radiusCellsX); i <= min(5, gridX + radiusCellsX); i++) {
    for (int j = max(0, gridY - radiusCellsY); j <= min(5, gridY + radiusCellsY); j++) {
      amp[j]++;
    }
  }
}

void drawToneMatrix() {
  int cellWidth = width / 6;
  int cellHeight = height / 6;
  
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      int x = i * cellWidth;
      int y = j * cellHeight;
      stroke(#005102); 
      noFill();
      rect(x, y, cellWidth, cellHeight);
      fill(#05F215);
      textAlign(CENTER, CENTER);
      text(toneCell[j][i], x + cellWidth / 2, y + cellHeight / 2);
    }
  }
}

void sendOSCmessages() {
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      if (amp[j] > 0) {
        println("TONE VALUE : " + toneCell[j][i] + " AMP : " + amp[j]);
        OscMessage msg1 = new OscMessage("/PlayS");
        msg1.add(toneCell[j][i]);
        msg1.add(map(amp[j], 0, cellCount / 6, 0, 1));
        osc.send(msg1, supercollider); 
      }
    }
  }
}

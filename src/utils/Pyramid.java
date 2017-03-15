package utils;

import org.opencv.core.Mat;

public class Pyramid {
	int windowSizeC;
	int windowSizeR;
	double windowSizeIncrement;
	int stepSize;
	Mat image;
	
	public Pyramid (Mat image, int windowSizeC, int windowSizeR, double windowSizeIncrement,int stepSize){
//		this.image = image;
		this.windowSizeC = windowSizeC;
		this.windowSizeR = windowSizeR;
		this.windowSizeIncrement = windowSizeIncrement;
		this.stepSize = stepSize;
				
//		pyramidStructure(image);
	}
	
	public void setRowWindowSize(int size){
		this.windowSizeR = size;
	}
	
	public int getRowWindowSize(){
		return this.windowSizeR;	
	}
	
	public void setColumnWindowSize(int size){
		this.windowSizeC = size;
	}
	
	public int getColumnWindowSize(){
		return this.windowSizeC;
	}
	
	public int getTestColumnWindowSize(){
		return (int)(windowSizeC * windowSizeIncrement);
	}
	
	public int getTestRowWindowSize(){
		return (int)(windowSizeR * windowSizeIncrement);
	}
}

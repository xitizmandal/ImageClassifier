package preprocessing;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.objdetect.HOGDescriptor;

public class Hog {
	private HOGDescriptor hogDescriptor;
	
	//Hog parameters
	int nbins = 9;
	int derivAperture = 0;
	int winSigma = -1;
	int histogramNormType = 0;
	double L2HysThreshold = 0.2;
	boolean gammaCorrection = false;
	int nlevels = 64;
	
	Size winSize;
	Size blockSize;
	Size blockStride;
	Size cellSize;
	Size winStride;
	Size padding;
	
	MatOfFloat descriptors;
	MatOfPoint locations;
	MatOfPoint foundLocations;
	MatOfDouble weights;
	
	public Hog(){
		winSize = new Size(128,64);
		blockSize = new Size(16,16);
		blockStride = new Size(8,8);
		cellSize = new Size(16,16);
		
		hogDescriptor = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
				derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels);
		
		winStride = new Size(16,16);
		padding = new Size(0,0);
		descriptors = new MatOfFloat(0);
		locations = new MatOfPoint();
		foundLocations = new MatOfPoint();
		weights = new MatOfDouble(0);
	}
	
	public void computeHog(Mat img){
		hogDescriptor.compute(img, descriptors, winStride, padding, locations);
//		hogDescriptor.detect(img, foundLocations, weights);
	}
	
	public long getDescriptorSize(){
		return hogDescriptor.getDescriptorSize();
	}
	
	public Mat getDescriptors(){
		return descriptors;
	}

}

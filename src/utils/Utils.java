package utils;

import java.io.File;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import preprocessing.Hog;

public class Utils {
	public static final int STANDARD_COMPUTE_SIZE = 128;
	public static final int NUMBER_OF_CLASSIFICATION = 18;
	public static final String CLASSIFIER_XML_ONEVSALL= "resource/Classifiers/OneVsAll/oneVsAll_";
	public static final String CLASSIFIER_XML_ONEVSONE = "resource/Classifiers/OneVsOne/oneVsOne_classifier.xml";
//	public static final String CLASSIFIER_XML_NAME = "resource/Classifiers_18/oneVsMultiple_";
//	public static final String CLASSIFIER_XML_ALL = "resource/Classifiers/";
	public static final String TRAIN_IMAGE_PATH = "resource/Train/sample_";
	public static final String OUTPUT_PATH = "resource/output/";
	public static final String TEST_PATH = "resource/newtests/";
	
	public static final String[] classification = {
			"crab",
			"panda",
			"croc",
			"dolphin",
			"dragonfly",
			"guitar",
			"elephant",
			"kangaroo",
			"leopard",
			"lobster",
			"saxaphone",
			"garfield",
			"motorcycle",
			"pizza",
			"strawberry",
			"scissors",
			"watch",
			"others"
			};
	
	public static Mat getMat( String path ) {
	    Mat img = new Mat();
	    Mat con = Highgui.imread( path, Highgui.CV_LOAD_IMAGE_GRAYSCALE );
	    con.convertTo(img, CvType.CV_32FC1, 1.0 / 255.0 );
	    return img;
	}
	
	public static Mat loadAndResizeImage(File file){
		Mat img = getMat(file.getAbsolutePath());
		return loadAndResizeImage(img);	
	}
	
	public static Mat loadAndResizeImage(String path){
		Mat img = getMat(path);
		return loadAndResizeImage(img);
	}
	
	public static Mat loadAndResizeImage(Mat mat){
		Imgproc.resize(mat, mat, new Size(STANDARD_COMPUTE_SIZE,STANDARD_COMPUTE_SIZE));
		return getHog(mat);
	}
		
	public static Mat getHog(Mat image){
		Hog hog = new Hog();
		image.convertTo(image, CvType.CV_8U);
		hog.computeHog(image);
		return hog.getDescriptors();		
	}
	
	public static String returnValString(float val){
		String toReturn = new String();
		switch((int) val){
			case 0:
				toReturn = classification[0];
				break;
				
			case 1:
				toReturn = classification[1];
				break;
				
			case 2:
				toReturn = classification[2];
				break;
				
			case 3:
				toReturn = classification[3];
				break;
				
			case 4:
				toReturn = classification[4];
				break;
				
			case 5:
				toReturn = classification[5];
				break;
				
			case 6:
				toReturn = classification[6];
				break;
				
			case 7:
				toReturn = classification[7];
				break;
				
			case 8:
				toReturn = classification[8];
				break;
				
			case 9:
				toReturn = classification[9];
				break;
			
			case 10:
				toReturn = classification[10];
				break;
				
			case 11:
				toReturn = classification[11];
				break;
				
			case 12:
				toReturn = classification[12];
				break;
				
			case 13:
				toReturn = classification[13];
				break;
				
			case 14:
				toReturn = classification[14];
				break;
				
			case 15:
				toReturn = classification[15];
				break;
				
			case 16:
				toReturn = classification[16];
				break;
			
			case 17:
				toReturn = classification[17];
				break;
		}
		return toReturn;
	}
	
	public static void checkDirectories(String path){
		File directory = new File(path);
		if(!directory.exists()){
			 System.out.println("creating directory: " + directory.getName());
			    boolean result = false;

			    try{
			        directory.mkdir();
			        result = true;
			    } 
			    catch(SecurityException se){
			        //handle it
			    }        
			    if(result) {    
			        System.out.println("DIR created");  
			    }
		} else {
			System.out.println("Present directory: " + directory.getName());
		}
		
	}
	
	public static void checkAllDirectories(){
		checkDirectories("resource");
		checkDirectories("resource/Classifiers");
		checkDirectories("resource/Classifiers/OneVsAll");
		checkDirectories("resource/Classifiers/OneVsOne");
		checkDirectories("resource/Train");
		checkDirectories(Utils.TEST_PATH);
		checkDirectories(Utils.OUTPUT_PATH);
	}
}

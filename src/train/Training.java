package train;

import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

import utils.Utils;

public class Training {
	public static final String TRAIN_IMAGE_PATH = "resource/Train/sample_";

	
	private int numberOfClassification;
	
	private Mat trainImages;
	private Mat trainLabels;
	private Mat trainData;
	private Mat classes;
	
	private int[] range;
	private int numberOfImages;
	private CvSVM classifier;
	
    CvSVMParams params;
	
	public Training(int numberOfClassification){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		this.numberOfClassification = numberOfClassification;
	}
	
	private void init(){
		trainImages = new Mat();
//		trainLabels = new Mat();
		trainData =  new Mat();
		classes = new Mat();
		range = new int[numberOfClassification+1];
		range[0] = 0;
		numberOfImages = 0;
	}
	
	private void loadSingle(String path, int index){
		for (File file: new File(path).listFiles()){
			trainImages.push_back(Utils.loadAndResizeImage(file).reshape(1,1));
			numberOfImages++;
		}
		range[index] = numberOfImages;
	}
	
	private void loadAll(){
		for (int i = 1; i <= numberOfClassification; i++){
			String path;
			if(i<10){
				path = TRAIN_IMAGE_PATH+"0"+i+"/";
			} else {
				path = TRAIN_IMAGE_PATH+i+"/";
			}
			loadSingle(path,i);
		}
	}
	
	private void setLabels(int sampleNo){
		
		for (int i = 0; i < numberOfClassification; i++){
			int labelValue = 0;
			if( i == sampleNo){
				labelValue = 1;
			}
			trainLabels.rowRange(range[i],range[i+1]).setTo(new Scalar(labelValue));
		}
		
	}
	
	private void setLabels(){
		for (int i = 0; i<numberOfClassification; i++){
			trainLabels.rowRange(range[i],range[i+1]).setTo(new Scalar(i));
		}
	}
		
	public void train(int sampleNo) {
		init();
		loadAll();
	    trainImages.copyTo( trainData );
	    trainData.convertTo( trainData, CvType.CV_32FC1 );
	    trainLabels = new Mat(numberOfImages,1,CvType.CV_32FC1);
	    
	    setLabels(sampleNo);

	    trainLabels.copyTo( classes );
	    System.out.println(Utils.classification[sampleNo]);
	    System.out.println("trainImages: " + trainImages);  
	    System.out.println("trainLabels: " + trainLabels);
	    
	    setCvSVMParams();
	    
	    classifier = new CvSVM( trainData, classes, new Mat(), new Mat(), params );
	    classifier.save(Utils.CLASSIFIER_XML_ONEVSALL+ Utils.classification[sampleNo] + ".xml");
	    
	}
	
	public void train() {
		init();
		loadAll();
	    trainImages.copyTo( trainData );
	    trainData.convertTo( trainData, CvType.CV_32FC1 );
	    trainLabels = new Mat(numberOfImages,1,CvType.CV_32FC1);
	    
	    setLabels();

	    trainLabels.copyTo( classes );
	    System.out.println("ALL");
	    System.out.println("trainImages: " + trainImages);  
	    System.out.println("trainLabels: " + trainLabels);
	    
	    setCvSVMParams();
	    
	    classifier = new CvSVM( trainData, classes, new Mat(), new Mat(), params );
	    classifier.save(Utils.CLASSIFIER_XML_ONEVSONE);
	}
	
	void setCvSVMParams(){
		params = new CvSVMParams();
	    params.set_kernel_type( CvSVM.POLY);
	    params.set_svm_type(CvSVM.C_SVC);
	    params.set_degree(2);
//	    params.set_gamma(gamma);
	    params.set_term_crit(new TermCriteria(TermCriteria.COUNT+TermCriteria.EPS,1000,10e-6));
	}
	
	

}

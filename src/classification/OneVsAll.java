package classification;

import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

import classifiers.OneVsAllClassifier;
import utils.Utils;

public class OneVsAll {
	private Mat trainImages;
	private Mat trainLabels;
	private Mat trainData;
	private Mat classes;
	
	private int[] range;
	private int numberOfImages;
	private CvSVM classifier;
	
	public OneVsAll(){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		classifier = new CvSVM();
		
	}
	
	private void init(){
		trainImages = new Mat();
//		trainLabels = new Mat();
		trainData =  new Mat();
		classes = new Mat();
		range = new int[Utils.NUMBER_OF_CLASSIFICATION+1];
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
		for (int i = 1;i <=Utils.NUMBER_OF_CLASSIFICATION;i++){
			String path;
			if(i<10){
				path = Utils.TRAIN_IMAGE_PATH+"0"+i+"/";
			} else {
				path = Utils.TRAIN_IMAGE_PATH+i+"/";
			}
			loadSingle(path,i);
		}
	}
		
	public void train(int sampleNo) {
		init();
		loadAll();
	    trainImages.copyTo( trainData );
	    trainData.convertTo( trainData, CvType.CV_32FC1 );
	    trainLabels = new Mat(numberOfImages,1,CvType.CV_32FC1);
	    
	    for (int i = 0; i < Utils.NUMBER_OF_CLASSIFICATION; i++){
	    	if (i == sampleNo){
	    		trainLabels.rowRange(range[i],range[i+1]).setTo(new Scalar(1));
	    	} else {
	    		trainLabels.rowRange(range[i],range[i+1]).setTo(new Scalar(0));
	    	}
	    }

	    trainLabels.copyTo( classes );
	    System.out.println(Utils.classification[sampleNo]);
	    System.out.println("trainImages: " + trainImages);  
	    System.out.println("trainLabels: " + trainLabels);
	    
	    CvSVMParams params = new CvSVMParams();
	    params.set_kernel_type( CvSVM.POLY);
	    params.set_svm_type(CvSVM.C_SVC);
	    params.set_degree(2);
	    params.set_term_crit(new TermCriteria(TermCriteria.COUNT+TermCriteria.EPS,1000,10e-6));
	    
	    classifier = new CvSVM( trainData, classes, new Mat(), new Mat(), params );
	    classifier.save(Utils.CLASSIFIER_XML_ONEVSALL + Utils.classification[sampleNo] + ".xml");
	}
	
	public void test(String filename, int sampleNo) {
		OneVsAllClassifier oneVsAllClassifier = new OneVsAllClassifier(sampleNo);
		Mat out = new Mat();
		out = Utils.loadAndResizeImage(Utils.TEST_PATH + filename);
		
		System.out.println(Utils.classification[sampleNo] + ":\t" + oneVsAllClassifier.getClassifier().predict(out)
				+ ":\t" + oneVsAllClassifier.getClassifier().predict(out,true));
	}

}

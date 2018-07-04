package client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import processHipo.DataProcess;
import utils.DomainUtils;

public class ClassifyRealData {

	public static void main(String[] args) throws IOException {

		String dir = DomainUtils.getDataLocation();
		List<String> aList = new ArrayList<>();
		aList.add(dir + "out_clas_003923.evio.80.hipo");
		aList.add(dir + "out_clas_003923.evio.8.hipo");

		// aList.add(dir + "out_clas_003923.evio.8.hipo");
		DataProcess dataProcess = new DataProcess(aList);
		dataProcess.processFile();
		// dataProcess.plotData();
		String fileName = "models/cnn_simpleMKfix.zip";
		FaultClassifier fClassifier = new FaultClassifier(fileName);

		// for (int i = 0; i < 100; i++) {
		// FaultFactory factory = new FaultFactory();
		// factory.getFault(1);
		//
		// // factory.plotData();
		// // int[] predictedClasses =
		// // fClassifier.predict(factory.getFeatureVector());
		// // System.out.println(Arrays.toString(predictedClasses) + " actual
		// // label " + factory.getReducedFaultIndex());
		//
		// System.out.println("Actual label: " +
		// Arrays.toString(factory.getReducedLabel()));
		//
		// INDArray predictionsAtXYPoints =
		// fClassifier.output(factory.getFeatureVector());
		// System.out.println("Predicted label: " +
		// Arrays.toString(predictionsAtXYPoints.toIntVector()));
		// System.out.println("##############################");
		//
		// }

		// for (int i = 1; i < 7; i++) {
		// for (int j = 1; j < 7; j++) {
		// int[] predictedClasses =
		// fClassifier.predict(dataProcess.getFeatureVector(i, j));
		// System.out.println(Arrays.toString(predictedClasses));
		// dataProcess.plotData(i, j);
		// }
		// }

		// int[] predictedClasses =
		// fClassifier.predict(dataProcess.getFeatureVector(1, 2));
		// System.out.println(Arrays.toString(predictedClasses));
		// dataProcess.plotData(1, 2);
		// List<String> allClassLabels = new
		// ReducedFaultRecordReader().getLabels();
		// String modelPrediction = allClassLabels.get(predictedClasses[0]);
		// System.out.print("\n the model predicted " + modelPrediction +
		// "\n\n");

	}

}

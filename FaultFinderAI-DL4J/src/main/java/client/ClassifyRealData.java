package client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;

import processHipo.DataProcess;
import strategies.FaultRecordScalerStrategy;
import strategies.StandardizeMinMax;

public class ClassifyRealData {

	public static void main(String[] args) throws IOException {

		FaultRecordScalerStrategy strategy = new StandardizeMinMax(0.05);
		// String mkDir =
		// "/Users/michaelkunkel/WORK/CLAS/CLAS12/CLAS12Data/RGACooked/V5b.2.1/";

		String mkDir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";// DomainUtils.getDataLocation();
		List<String> aList = new ArrayList<>();
		aList.add(mkDir + "out_clas_003923.evio.80.hipo");
		aList.add(mkDir + "out_clas_003923.evio.8.hipo");

		// aList.add(dir + "out_clas_003923.evio.8.hipo");
		DataProcess dataProcess = new DataProcess(aList);
		dataProcess.processFile();
		// dataProcess.plotData();
		String fileName = "models/SingleFaultClassifyingHVPinFault.zip";
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
		List<TCanvas> aTCanvas = new ArrayList<>();
		for (int i = 1; i < 7; i++) {
			System.out.println("IN LOOP");
			for (int j = 1; j < 7; j++) {
				// int j = 6;
				INDArray predictionsAtXYPoints = fClassifier.output(dataProcess.getFeatureVector(i, j, strategy));

				double[] predictedClasses = predictionsAtXYPoints.toDoubleVector();
				// int[] predictedClasses =
				// fClassifier.output(dataProcess.getFeatureVector(i, j,
				// strategy));

				predictedClasses = Arrays.stream(predictedClasses).map(x -> (x < 1E-04) ? 0.0 : x)
						.map(x -> Math.round(x * 10000.0) / 10000.0).toArray();
				System.out.println(Arrays.toString(predictedClasses) + "  S: " + i + " SL: " + j);
				System.out.println("################################################");

				dataProcess.plotData(i, j);
			}
		}

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

package client;

import java.io.IOException;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

import faulttypes.FaultFactory;

public class TestPredictions {

	public static void main(String[] args) throws IOException {
		String fileName = "models/testing.zip";// cnn_simpleMKfix.zip";
		FaultClassifier fClassifier = new FaultClassifier(fileName);
		int tPositive = 0;
		int fNegative = 0;
		for (int i = 0; i < 100; i++) {
			FaultFactory factory = new FaultFactory();
			factory.getFault(1);
			System.out.println("Actual label:    " + Arrays.toString(factory.getReducedLabel()));

			INDArray predictionsAtXYPoints = fClassifier.output(factory.getFeatureVector());
			int[] predictionArray = predictionsAtXYPoints.toIntVector();
			System.out.println("Predicted label: " + Arrays.toString(predictionArray));

			int predictionIndex = 0;
			int trueIndex = factory.getReducedFaultIndex();
			for (int j = 0; j < predictionArray.length; j++) {
				if (predictionArray[j] == 1) {
					predictionIndex = j;
				}
			}
			if ((predictionIndex - trueIndex) != 0) {
				fNegative++;
			} else {
				tPositive++;
			}
		}
		System.out.println(tPositive + "  " + fNegative);
		System.out.println("Recall is = " + (double) (tPositive / (tPositive + fNegative)));
	}
}

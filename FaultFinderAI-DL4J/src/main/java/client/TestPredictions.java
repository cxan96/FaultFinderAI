package client;

import java.io.IOException;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

import faulttypes.FaultFactory;

public class TestPredictions {

	public static void main(String[] args) throws IOException {
		String fileName = "models/cnn_simpleMKfix.zip";
		FaultClassifier fClassifier = new FaultClassifier(fileName);
		for (int i = 0; i < 100; i++) {
			FaultFactory factory = new FaultFactory();
			factory.getFault(1);
			System.out.println("Actual label:    " + Arrays.toString(factory.getReducedLabel()));

			INDArray predictionsAtXYPoints = fClassifier.output(factory.getFeatureVector());
			System.out.println("Predicted label: " + Arrays.toString(predictionsAtXYPoints.toIntVector()));
			System.out.println("##############################");

		}
	}
}

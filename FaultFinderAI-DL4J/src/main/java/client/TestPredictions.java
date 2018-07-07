package client;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import org.nd4j.linalg.api.ndarray.INDArray;

import faultrecordreader.FaultRecorderScaler;
import faulttypes.FaultFactory;
import utils.DomainUtils;

public class TestPredictions {

	public static void main(String[] args) throws IOException {
		String fileName = DomainUtils.getDropboxLocal() + "2018-07-06-10-38-16anotherTest.zip";// cnn_simpleMKfix.zip"
		// testing.zip;
		FaultClassifier fClassifier = new FaultClassifier(fileName);
		int tPositive = 0;
		int fNegative = 0;
		for (int i = 0; i < 1000; i++) {
			FaultFactory factory = new FaultFactory();
			int faultType = ThreadLocalRandom.current().nextInt(7);
			factory.getFault(faultType);
			// factory.plotData();
			System.out.println(faultType + "  Actual label: " + Arrays.toString(factory.getReducedLabel()));

			INDArray features = factory.getFeatureVector();
			// double maxRange = (double) features.maxNumber();
			// double minRange = (double) features.minNumber();
			// System.out.println(maxRange + " " + minRange);

			FaultRecorderScaler scaler = new FaultRecorderScaler();
			scaler.preProcess(features);

			//
			// features.divi((maxRange));

			INDArray predictionsAtXYPoints = fClassifier.output(features);

			double[] predictionArray = predictionsAtXYPoints.toDoubleVector();
			// System.out.println("Predicted label: " +
			// Arrays.toString(predictionArray));

			int predictionIndex = 0;
			double predictionMax = 0;

			int trueIndex = factory.getReducedFaultIndex();
			for (int j = 0; j < predictionArray.length; j++) {
				if (predictionArray[j] > predictionMax) {
					predictionIndex = j;
					predictionMax = predictionArray[j];
				}
			}
			if ((predictionIndex - trueIndex) != 0) {
				fNegative++;
			} else {
				tPositive++;
			}
			// System.out.println("##############################");

		}
		System.out.println(tPositive + "  " + fNegative);
		System.out.println("Recall is = " + ((double) tPositive / ((double) (tPositive + fNegative))));
	}
}

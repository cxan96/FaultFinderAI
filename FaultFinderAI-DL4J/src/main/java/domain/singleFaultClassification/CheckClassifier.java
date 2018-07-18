package domain.singleFaultClassification;

import java.io.IOException;

import org.deeplearning4j.eval.Evaluation;

import client.FaultClassifier;
import faultrecordreader.SingleFaultRecorder;
import faulttypes.FaultFactory;
import strategies.MinMaxStrategy;

public class CheckClassifier {
	public static void main(String[] args) throws IOException {
		int[] faultLabels = { 6 };// 0, 1, 2, 3, 5
		// int i = 4;
		for (int i = 0; i < faultLabels.length; i++) {
			FaultFactory factory = new FaultFactory();
			factory.getFault(faultLabels[i]);
			System.out.println(factory.getFaultName());
			String fileName = "models/" + factory.getFaultName() + ".zip";
			FaultClassifier classifier = new FaultClassifier(fileName);
			int nLabels = factory.getFaultLabel().length;

			Evaluation evaluation = classifier.evaluate(nLabels, 1, 10000, new SingleFaultRecorder(faultLabels[i]),
					new MinMaxStrategy());
			System.out.println(evaluation.stats());
		}

	}

}

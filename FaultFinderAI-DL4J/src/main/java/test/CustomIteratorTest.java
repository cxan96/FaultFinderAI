package test;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * This class is for testing the custom XorDataSetIterator.
 */
public class CustomIteratorTest {
	public static void main(String args[]) {

		// Get the custom DataSetIterators:
		DataSetIterator xorTrain = new XorDataSetIterator();
		DataSetIterator xorTest = new XorDataSetIterator();

		System.out.println("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123).updater(new Nesterovs(0.006, 0.9))
				.l2(1e-4).list()
				.layer(0,
						new DenseLayer.Builder().nIn(2).nOut(5).activation(Activation.RELU)
								.weightInit(WeightInit.XAVIER).build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create output layer
						.nIn(5).nOut(2).activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		// print the score with every 1iteration
		model.setListeners(new ScoreIterationListener(1));

		System.out.println("Train model....");
		for (int i = 0; i < 500; i++) {
			model.fit(xorTrain);
		}

		System.out.println("Evaluate model....");
		Evaluation eval = new Evaluation(2);
		while (xorTest.hasNext()) {
			DataSet next = xorTest.next();
			INDArray output = model.output(next.getFeatures()); // get the networks prediction
			eval.eval(next.getLabels(), output); // check the prediction against the true class
		}

		System.out.println(eval.stats());

		System.exit(0);
	}
}

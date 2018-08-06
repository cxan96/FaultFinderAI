package domain.singleFaultClassification;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import client.ModelFactory;
import faultrecordreader.FaultRecorderScaler;
import faultrecordreader.KunkelPetersFaultRecorder;
import faults.FaultNames;
import strategies.MinMaxStrategy;

public class SideApproach2 {

	public static void main(String[] args) throws IOException {
		int faultType = 6;
		MultiLayerNetwork net = ModelFactory.deeperCNN(2);
		// set up a local web-UI to monitor the training available at
		// localhost:9000
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();

		net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1000));
		uiServer.attach(statsStorage);
		for (int i = 0; i < 3; i++) {

			DataSetIterator iter = new RecordReaderDataSetIterator.Builder(
					new KunkelPetersFaultRecorder(1, 10, FaultNames.CHANNEL_ONE, false), 1)
							// currently there are 14 labels in the dataset
							.classification(1, 2).maxNumBatches(112 * 6)
							.preProcessor(new FaultRecorderScaler(new MinMaxStrategy())).build();

			List<INDArray> featuresTrain = new ArrayList<>();
			List<INDArray> labelsTrain = new ArrayList<>();
			List<INDArray> labelsTest = new ArrayList<>();

			Random r = new Random(12345);
			while (iter.hasNext()) {
				DataSet ds = iter.next();
				featuresTrain.add(ds.getFeatureMatrix());
				labelsTrain.add(ds.getLabels());
				INDArray indexes = Nd4j.argMax(ds.getLabels(), 1); // Convert
																	// from
																	// one-hot
																	// representation
																	// -> index
				labelsTest.add(indexes);
			}
			// System.out.println("F size: " + featuresTrain.size() + " l size:
			// " + labelsTrain.size());
			// System.out.println(labelsTest.get(0));
			// Train model:
			int nEpochs = 25;
			for (int epoch = 0; epoch < nEpochs; epoch++) {
				for (int j = 0; j < featuresTrain.size(); j++) {
					net.fit(featuresTrain.get(j), labelsTrain.get(j));

				}

				// System.out.println("Epoch " + epoch + " complete");
			}
			iter.reset();
		}
		ModelSerializer.writeModel(net, new File("models/testTest.zip"), false);

		DataSetIterator iterTest = new RecordReaderDataSetIterator.Builder(
				new KunkelPetersFaultRecorder(1, 10, FaultNames.CHANNEL_ONE, false), 1)
						// currently there are 14 labels in the dataset
						.classification(1, 2).maxNumBatches(10000)
						.preProcessor(new FaultRecorderScaler(new MinMaxStrategy())).build();
		Evaluation evaluation = net.evaluate(iterTest);
		System.out.println(evaluation.stats(false, true));

	}

}

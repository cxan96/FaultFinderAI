package test;

import java.util.List;

import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import faultrecordreader.FaultRecordReader;
import faultrecordreader.FaultRecorderScaler;
import utils.ArrayUtilities;

/**
 * Jefferson Laboratory Hall-B CLAS12 Drift Chamber Fault Classification --
 * ----------------- There is much room for improvement here -----------------
 * a) Currently we are interested in the faults and their locations, but some
 * faults are the same graphically i.e. a hvPinFault bundle graphically has the
 * same shape in all 6 of its layers. So we could devise a easier CNN to just
 * look at 12 faults in hvPinFault instead of including all the layers
 * 
 * Example classification of photos from 1487 different faults; i.e.
 * 
 * hvPinFault - 72 different faults(12 faults grouped in wires bundles with each
 * wire bundle 6 layers high)
 * 
 * hvChannelFault - 8 different faults (8 bundles of wires extending all the
 * layers)
 * 
 * hvConnectorFault - 42 faults (the "E's, 3's and Trees"), basic repeating
 * pattern of wire bundles extending all 6 layers. Repeats every super bundle of
 * "E3Tree"
 *
 * hvFuseFault - 21 faults comprised of a combo of the "E3Tree" setup, also
 * repeating every 3 super-bundle
 * 
 * hvWireFault - 672 faults (112 wires * 6 layers). There are two types this -
 * 1) Dead wire - Wire does not fire/inefficient , no/limited statistics
 * 
 * 2) Hot wire - Wire is associated with too much noise and constantly fires,
 * initially defined as 50% more hit rate then surrounding wires
 *
 * CHALLENGE: Using the FaultRecordReader, which utilizes the factory method of
 * getting a fault, train single fault classification and the extend to multiple
 * fault classification, i.e. multiple faults per superlayer
 */
public class TestFaultRecordReader {

	private static final Logger log = LoggerFactory.getLogger(TestFaultRecordReader.class);
	private static int batchSize = 200;
	private static int testSize = 20;
	private static int MAX_BATCHES = 100000;

	private static long seed = 42;
	private static int epochs = 50;
	private double learningRate = 0.005;

	private static int numInputs = ArrayUtilities.nLayers * ArrayUtilities.nWires;
	private static int numHiddenNodes = ArrayUtilities.nWires;
	private static int numLabels = ArrayUtilities.faultLableSize;

	private DataNormalization scaler = null;
	private FaultRecordReader recordReader = null;
	private FaultRecordReader recordReaderTest = null;

	private DataSetIterator trainIter = null;
	private DataSetIterator testIter = null;

	private MultiLayerNetwork network = null;
	private MultiLayerConfiguration conf = null;
	private MultiLayerNetwork model = null;

	public TestFaultRecordReader() {
		this.scaler = new FaultRecorderScaler();

		initTrainingRecorder();
		initTestingRecorder();
		initNetwork();
		initModel();
	}

	private void initTrainingRecorder() {
		/**
		 * Data Setup -> define how to load data into net: - recordReader = the
		 * reader that loads the fault data pass
		 **/
		this.recordReader = new FaultRecordReader();
		trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels, MAX_BATCHES);

		scaler.fit(trainIter);
		trainIter.setPreProcessor(scaler);

	}

	private void initTestingRecorder() {
		/**
		 * Testing Setup -> define how to load data into net: - recordReader =
		 * the reader that loads the fault data pass
		 **/
		this.recordReaderTest = new FaultRecordReader();
		testIter = new RecordReaderDataSetIterator(recordReaderTest, testSize, 1, numLabels, MAX_BATCHES);

		scaler.fit(testIter);
		testIter.setPreProcessor(scaler);

	}

	private void initNetwork() {
		// log.info("Build model....");
		conf = new NeuralNetConfiguration.Builder().seed(seed).updater(new Nesterovs(learningRate, 0.9)).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(1,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER)
								.activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numLabels).build())
				.pretrain(false).backprop(true).build();
	}

	private void initModel() {
		this.model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(500)); // Print score
																// every 500
																// parameter
																// updates
	}

	public void train() {
		for (int n = 0; n < epochs; n++) {
			model.fit(trainIter);
		}

		System.out.println("Evaluate model....");
		Evaluation eval = new Evaluation(numLabels);
		while (testIter.hasNext()) {
			DataSet t = testIter.next();
			INDArray features = t.getFeatureMatrix();
			INDArray labels = t.getLabels();
			INDArray predicted = model.output(features, false);

			eval.eval(labels, predicted);
		}

		// Print the evaluation statistics
		System.out.println(eval.stats());
	}

	public static void main(String[] args) {
		TestFaultRecordReader test = new TestFaultRecordReader();
		test.train();
	}

	public static void oldMain(String[] args) {
		FaultRecordReader recordReader = new FaultRecordReader();
		List<Writable> aList = recordReader.next();
		System.out.println(aList.get(0).toString());

		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
		DataNormalization scaler = new FaultRecorderScaler();
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);
		DataSet dSet = dataIter.next(1);
		System.out.println("########### Features ################");

		System.out.println(dSet.getFeatures().toString());

		H1F aH1f = new H1F("a h1f", 100, 0, 1.1);
		System.out.println("dSet.getFeatures().length()  " + dSet.getFeatures().length());
		for (int i = 0; i < dSet.getFeatures().length(); i++) {
			double adub = dSet.getFeatures().getDouble(i);
			if (adub > 1) {
				System.out.println("MORE  " + adub);
			}
			if (adub < 1) {
				System.out.println("LESS  " + adub);
			}
			aH1f.fill(adub);
		}
		TCanvas canvas = new TCanvas("Canvas", 800, 800);
		canvas.draw(aH1f);
		System.out.println("######### Labels ##################");

		System.out.println(dSet.getLabels().toString());
		System.out.println("###########################");
		System.out.println("###########################");
		System.out.println(dSet.getLabels().length() + "   length");
		dSet.getLabels().toString();
		for (int i = 0; i < dSet.getLabels().length() - 1; i++) {
			int labelPos = dSet.getLabels().getInt(i);
			if (labelPos == 1) {
				System.out.println(" ****  " + dSet.getLabels().getInt(i) + "  " + i);
			}

		}
		System.out.println(recordReader.getLabelInt());

	}

	private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad,
			double bias) {
		return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
	}

	private ConvolutionLayer conv3x3(String name, int out, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(out).biasInit(bias).build();
	}

	private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 5, 5 }, stride, pad).name(name).nOut(out).biasInit(bias)
				.build();
	}

	private SubsamplingLayer maxPool(String name, int[] kernel) {
		return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 }).name(name).build();
	}

	private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
		return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
	}
}

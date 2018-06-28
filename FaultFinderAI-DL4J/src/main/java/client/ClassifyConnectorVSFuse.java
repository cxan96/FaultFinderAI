package client;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import faultrecordreader.FaultRecordReader;
import utils.ArrayUtilities;

public class ClassifyConnectorVSFuse {
	private static final Logger log = LoggerFactory.getLogger(ClassifyConnectorVSFuse.class);
	private static int batchSize = 200;
	private static int testSize = 20;
	private static int MAX_BATCHES = 1000;
	private static boolean save = true;

	private static long seed = 42;
	private static int epochs = 50;
	private double learningRate = 0.005;
	private int scoreIterations = 500;
	private static int numInputs = ArrayUtilities.nLayers * ArrayUtilities.nWires;
	private static int numHiddenNodes = ArrayUtilities.nWires;
	private static int numLabels = ArrayUtilities.hvConnectorFault.length + ArrayUtilities.hvFuseFault.length;

	private DataNormalization scaler = null;
	private FaultRecordReader recordReader = null;
	private FaultRecordReader recordReaderTest = null;

	private DataSetIterator trainIter = null;
	private DataSetIterator testIter = null;

	private MultiLayerNetwork network = null;
	private MultiLayerConfiguration conf = null;
	private MultiLayerNetwork model = null;
}

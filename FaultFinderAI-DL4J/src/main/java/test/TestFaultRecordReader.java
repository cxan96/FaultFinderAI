package test;

import java.util.Random;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import faultrecordreader.FaultRecordReader;

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

	protected static final Logger log = LoggerFactory.getLogger(TestFaultRecordReader.class);
	protected static int batchSize = 20;

	protected static long seed = 42;
	protected static Random rng = new Random(seed);
	protected static int epochs = 50;
	protected static double splitTrainTest = 0.8;
	protected static boolean save = false;
	protected static int maxPathsPerLabel = 18;

	protected static String modelType = "AlexNet"; // LeNet, AlexNet or Custom
													// but you need to fill it
													// out
	private int numLabels;

	public static void main(String[] args) {

		DataNormalization scaler = new ImagePreProcessingScaler(0, 1, 1);
		FaultRecordReader recordReader = new FaultRecordReader();
		DataSetIterator dataIter;

		log.info("Train model....");
		// Train without transformations
		recordReader.initialize(trainData, null);
		dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);
		network.fit(dataIter, epochs);
	}
}

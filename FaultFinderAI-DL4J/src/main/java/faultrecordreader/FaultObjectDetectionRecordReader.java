package faultrecordreader;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.datavec.image.util.ImageUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import faults.Fault;
import faults.FaultFactory;
import faults.FaultNames;

/**
 * An fault record reader for object detection.
 * <p>
 * Format of returned values: 4d array, with dimensions [minibatch, 4+C, h, w]
 * Where the image is quantized into h x w grid locations.
 * <p>
 * Note that this is a modified version of Alex Black's
 * ObjectDetectionRecordReader.java
 *
 * @author Michael C. Kunkel
 */
public class FaultObjectDetectionRecordReader implements RecordReader {

	private final int gridW;
	private final int gridH;
	private int height;
	private int width;
	private int channels;
	protected List<String> labels;

	protected FaultFactory factory = null;
	private int label;
	// args for FaultFactory constructor
	private int superLayer;
	private int maxFaults;
	private FaultNames desiredFault;
	private boolean singleFaultGeneration;
	private boolean blurredFaults;

	/**
	 *
	 * @param height        Height of the output images
	 * @param width         Width of the output images
	 * @param channels      Number of channels for the output images
	 * @param gridH         Grid/quantization size (along height dimension) - Y axis
	 * @param gridW         Grid/quantization size (along height dimension) - X axis
	 * @param labelProvider ImageObjectLabelProvider - used to look up which objects
	 *                      are in each image
	 */
	public FaultObjectDetectionRecordReader(int superLayer, int maxFaults, FaultNames desiredFault,
			boolean singleFaultGeneration, boolean blurredFaults, int height, int width, int channels, int gridH,
			int gridW) {
		this.superLayer = superLayer;
		this.maxFaults = maxFaults;
		this.desiredFault = desiredFault;
		this.singleFaultGeneration = singleFaultGeneration;
		this.blurredFaults = blurredFaults;
		this.factory = new FaultFactory(superLayer, maxFaults, desiredFault, singleFaultGeneration, blurredFaults,
				channels);

		this.height = height;
		this.width = width;
		this.channels = channels;
		this.gridW = gridW;
		this.gridH = gridH;
		initialize();
	}

	private void initialize() {
		Set<String> labelSet = new HashSet<>();
		/**
		 * OK, we need all the faults loaded at once otherwise it doesn't make sense
		 * with the one-hot representation
		 */
		for (FaultNames d : FaultNames.values()) {
			labelSet.add(d.getSaveName());
		}
		labels = new ArrayList<>(labelSet);

		// To ensure consistent order for label assignment (irrespective of file
		// iteration order), we want to sort the list of labels
		Collections.sort(labels);
	}

	// MK Testing stuff

	public FaultFactory getFactory() {
		return factory;
	}

	// End MK testing stuff
	@Override
	public boolean batchesSupported() {
		// I might want to set this to true so that I train in batches, reduces
		// memory
		// will get back to this after impl of modes
		return true;
	}

	@Override
	public boolean hasNext() {
		// since this is batch mode, and the data is generated on the fly, this
		// should always be true
		return true;
	}

	@Override
	public List<Writable> next() {
		return next(1).get(0);
	}

	@Override
	public void reset() {
		this.factory = new FaultFactory(this.superLayer, this.maxFaults, this.desiredFault, this.singleFaultGeneration,
				this.blurredFaults, this.channels);
	}

	@Override
	public boolean resetSupported() {
		// Why would we need to reset in this type of training?
		return true;
	}

	@Override
	public List<List<Writable>> next(int num) {
		List<INDArray> faultData = new ArrayList<>(num);
		List<List<Fault>> objects = new ArrayList<>(num);

		for (int i = 0; i < num && hasNext(); i++) {
			faultData.add(factory.getFeatureVectorAsMatrix());
			// faultData.add(factory.getFeatureVector());

			objects.add(factory.getFaultList());

		}

		int nClasses = labels.size();

		INDArray outImg = Nd4j.create(faultData.size(), channels, height, width);
		INDArray outLabel = Nd4j.create(faultData.size(), 4 + nClasses, gridH, gridW);

		int exampleNum = 0;
		for (int i = 0; i < faultData.size(); i++) {
			INDArray imageFile = faultData.get(i);

			outImg.put(new INDArrayIndex[] { point(exampleNum), all(), all(), all() }, imageFile);

			List<Fault> objectsThisImg = objects.get(exampleNum);

			label(imageFile, objectsThisImg, outLabel, exampleNum);

			exampleNum++;
		}
		reset();
		// System.out.println("#########" + outLabel.shapeInfoToString() + " " +
		// objects.size());
		return new NDArrayRecordBatch(Arrays.asList(outImg, outLabel));
	}

	private void label(INDArray image, List<Fault> objectsThisImg, INDArray outLabel, int exampleNum) {
		int oW = image.columns(); // should be 6
		int oH = image.rows(); // should be 112

		int W = oW;
		int H = oH;
		// put the label data into the output label array
		for (Fault io : objectsThisImg) {
			/**
			 * OK here is a little nuance. The locations of the Faults are in natural CLAS
			 * x->wires; y->layers coordinates. But the featured data itself is in
			 * columns->x->layers; rows->y->wires so we should SWITCH XCenter <-> YCenter
			 * XMin <-> YMin XMax <-> YMax
			 * 
			 */

			double cx = io.getFaultCoordinates().getYCenterPixels();
			double cy = io.getFaultCoordinates().getXCenterPixels();
			double[] cxyPostScaling = ImageUtils.translateCoordsScaleImage(cx, cy, W, H, width, height);
			double[] tlPost = ImageUtils.translateCoordsScaleImage(io.getFaultCoordinates().getYMin(),
					io.getFaultCoordinates().getXMin(), W, H, width, height);
			double[] brPost = ImageUtils.translateCoordsScaleImage(io.getFaultCoordinates().getYMax(),
					io.getFaultCoordinates().getXMax(), W, H, width, height);

			// Get grid position for image
			int imgGridX = (int) (cxyPostScaling[0] / width * gridW);
			int imgGridY = (int) (cxyPostScaling[1] / height * gridH);

			// Convert pixels to grid position, for TL and BR X/Y
			tlPost[0] = tlPost[0] / width * gridW;
			tlPost[1] = tlPost[1] / height * gridH;
			brPost[0] = brPost[0] / width * gridW;
			brPost[1] = brPost[1] / height * gridH;

			// MK Debugging
			// System.out.println(" oW " + oW + " oH " + oH + " W " + W + " H "
			// + H);
			if (imgGridY > 55) {
				factory.draw();
				System.out.println(io.getSubFaultName());
				System.out.println(" cx " + cx + " cy " + cy);
				System.out.println("io.getFaultCoordinates().getyMin() " + io.getFaultCoordinates().getYMin()
						+ " io.getFaultCoordinates().getxMin() " + io.getFaultCoordinates().getXMin());
				System.out.println("io.getFaultCoordinates().getyMax() " + io.getFaultCoordinates().getYMax()
						+ " io.getFaultCoordinates().getxMax() " + io.getFaultCoordinates().getXMax());
				System.out.println(exampleNum + "  " + 0 + "   " + imgGridY + "   " + imgGridX + "  " + tlPost[0]
						+ "   " + width + "  " + gridW + "  " + height + "  " + gridH);
			}
			// Put TL, BR into label array:
			outLabel.putScalar(exampleNum, 0, imgGridY, imgGridX, tlPost[0]);
			outLabel.putScalar(exampleNum, 1, imgGridY, imgGridX, tlPost[1]);
			outLabel.putScalar(exampleNum, 2, imgGridY, imgGridX, brPost[0]);
			outLabel.putScalar(exampleNum, 3, imgGridY, imgGridX, brPost[1]);

			// Put label class into label array: (one-hot representation)
			int labelIdx = labels.indexOf(io.getSubFaultName().getSaveName());
			outLabel.putScalar(exampleNum, 4 + labelIdx, imgGridY, imgGridX, 1.0);
		}
	}

	@Override
	public Record nextRecord() {
		List<Writable> list = next();
		URI uri = URI.create("FaultFinderAI");
		// return new org.datavec.api.records.impl.Record(list, metaData)
		return new org.datavec.api.records.impl.Record(list,
				new RecordMetaDataURI(null, FaultObjectDetectionRecordReader.class));

	}

	// the rest below here are not needed, but kept for as need

	@Override
	public void close() throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public Configuration getConf() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setConf(Configuration arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<String> getLabels() {
		return this.labels;
	}

	@Override
	public List<RecordListener> getListeners() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initialize(InputSplit arg0) throws IOException, InterruptedException {
		// TODO Auto-generated method stub

	}

	@Override
	public void initialize(Configuration arg0, InputSplit arg1) throws IOException, InterruptedException {
		// TODO Auto-generated method stub

	}

	@Override
	public Record loadFromMetaData(RecordMetaData arg0) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Record> loadFromMetaData(List<RecordMetaData> arg0) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Writable> record(URI arg0, DataInputStream arg1) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setListeners(RecordListener... arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setListeners(Collection<RecordListener> arg0) {
		// TODO Auto-generated method stub

	}

}
package faultrecordreader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.nd4j.linalg.api.concurrency.AffinityManager;
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
 * ObjectDetectionRecordReader
 *
 * @author Michael C. Kunkel
 */
public class FaultObjectDetectionRecordReader extends KunkelPetersFaultRecorder {

	private final int gridW;
	private final int gridH;
	private int height;
	private int width;
	private int channels;

	protected FaultFactory factory = null;
	private int label;
	// args for FaultFactory constructor
	private int superLayer;
	private int maxFaults;
	private FaultNames desiredFault;
	private boolean singleFaultGeneration;

	/**
	 *
	 * @param height
	 *            Height of the output images
	 * @param width
	 *            Width of the output images
	 * @param channels
	 *            Number of channels for the output images
	 * @param gridH
	 *            Grid/quantization size (along height dimension) - Y axis
	 * @param gridW
	 *            Grid/quantization size (along height dimension) - X axis
	 * @param labelProvider
	 *            ImageObjectLabelProvider - used to look up which objects are
	 *            in each image
	 */
	public FaultObjectDetectionRecordReader(int superLayer, int maxFaults, FaultNames desiredFault,
			boolean singleFaultGeneration, boolean blurredFaults, int height, int width, int channels, int gridH,
			int gridW) {
		super(superLayer, maxFaults, desiredFault, singleFaultGeneration, blurredFaults);
		this.height = height;
		this.width = width;
		this.channels = channels;
		this.gridW = gridW;
		this.gridH = gridH;

	}

	@Override
	public List<List<Writable>> next(int num) {
		List<Fault> faults = new ArrayList<>(num);
		List<List<Fault>> objects = new ArrayList<>(num);

		for (int i = 0; i < num && hasNext(); i++) {
			File f = iter.next();
			this.currentFile = f;
			if (!f.isDirectory()) {
				files.add(f);
				objects.add(labelProvider.getImageObjectsForPath(f.getPath()));
			}
		}

		int nClasses = labels.size();

		INDArray outImg = Nd4j.create(files.size(), channels, height, width);
		INDArray outLabel = Nd4j.create(files.size(), 4 + nClasses, gridH, gridW);

		int exampleNum = 0;
		for (int i = 0; i < files.size(); i++) {
			File imageFile = files.get(i);
			this.currentFile = imageFile;
			try {
				this.invokeListeners(imageFile);
				Image image = this.imageLoader.asImageMatrix(imageFile);
				this.currentImage = image;
				Nd4j.getAffinityManager().ensureLocation(image.getImage(), AffinityManager.Location.DEVICE);

				outImg.put(new INDArrayIndex[] { point(exampleNum), all(), all(), all() }, image.getImage());

				List<ImageObject> objectsThisImg = objects.get(exampleNum);

				label(image, objectsThisImg, outLabel, exampleNum);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}

			exampleNum++;
		}

		return new NDArrayRecordBatch(Arrays.asList(outImg, outLabel));
	}
}
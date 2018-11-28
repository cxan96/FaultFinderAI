/**
 * 
 */
package test;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.util.List;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import clasDC.faults.FaultNames;
import domain.objectDetection.FaultObjectClassifier;
import faultrecordreader.FaultObjectDetectionImageRecordReader;
import faultrecordreader.FaultRecorderScaler;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
public class DrawFaults {
	private DataSetIterator test;
	private FaultObjectClassifier classifier;
	// private FaultObjectDetectionRecordReader recordReader;
	private FaultObjectDetectionImageRecordReader recordReader;

	public DrawFaults() {
		initialize();
	}

	private void initialize() {
//		this.recordReader = new FaultObjectDetectionRecordReader(3, 10, FaultNames.CHANNEL_ONE, true, true, 112, 6, 3,
//				56, 3);
		this.recordReader = new FaultObjectDetectionImageRecordReader(3, 10, FaultNames.CHANNEL_ONE, true, true, 6, 112,
				3, 3, 28);
		FaultRecordScalerStrategy strategy = new MinMaxStrategy();
		// FaultRecordScalerStrategy strategy = new IdentityStrategy();

		this.test = new RecordReaderDataSetIterator.Builder(recordReader, 1).regression(1).maxNumBatches(1)
				.preProcessor(new FaultRecorderScaler(strategy)).build();
	}

	public FaultObjectClassifier getClassifier() {
		return this.classifier;
	}

	public DataSetIterator getDSIterator() {
		return this.test;
	}

//	public FaultObjectDetectionRecordReader getRecordReader() {
//		return this.recordReader;
//	}
	public FaultObjectDetectionImageRecordReader getRecordReader() {
		return this.recordReader;
	}

	public static void main(String[] args) throws Exception {
		int gridWidth = 6;
		int gridHeight = 112;

		DrawFaults drawFaults = new DrawFaults();
		FaultObjectDetectionImageRecordReader rrTransform = drawFaults.getRecordReader();
		List<Writable> next = rrTransform.next();
		INDArray labelArray = ((NDArrayWritable) next.get(1)).get();

		RecordReaderDataSetIterator test = (RecordReaderDataSetIterator) drawFaults.getDSIterator();
		org.nd4j.linalg.dataset.DataSet ds = test.next();

//		List<String> labels = test.getLabels();
//		for (String string : labels) {
//			System.out.println(string);
//		}
		// RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)
		// ds.getExampleMetaData().get(0);
		NativeImageLoader imageLoader = new NativeImageLoader();
		CanvasFrame frame = new CanvasFrame("Test");
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

		INDArray features = ds.getFeatures();
		INDArray labels = ds.getLabels();
		int rank = labels.rank();
		int rows = (int) labels.size(rank == 3 ? 1 : 2);
		int cols = (int) labels.size(rank == 3 ? 2 : 3);
		int one = (int) labels.size(0);
		int two = (int) labels.size(1);
		int size1 = (int) labels.size(1);
		int c = (int) labels.size(1) - 4;
		int[] nhw = new int[] { 1, 3, 28 };

//		for (int i = 0; i < two; i++) {
//			System.out.println(labels.getRow(0).getScalar(2) + "   test");
//
//			System.out.println(labels.getInt(0, i, 0, 0) + "  " + labels.getInt(1, i, 0, 0) + "  "
//					+ labels.getInt(2, i, 0, 0) + "  " + labels.getInt(3, i, 0, 0) + "  " + labels.getInt(4, i, 0, 0));
//
//		}
		// drawFaults.loadImage(features);
		FaultUtils.draw(features);
		Mat mat = imageLoader.asMat(features);
		Mat convertedMat = new Mat();
		mat.convertTo(convertedMat, 0, 255, 0);

		// mat.convertTo(convertedMat, 6, 255, 0);

		int w = 112 * 4;
		int h = 6 * 100;
		Mat image = new Mat();

		resize(convertedMat, image, new Size(w, h));

		frame.setTitle(" - test");
		frame.setCanvasSize(w, h);
		frame.showImage(converter.convert(convertedMat));
		// FaultUtils.draw(FaultUtils.dataSector1);

	}
}

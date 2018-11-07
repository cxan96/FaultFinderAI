/**
 * 
 */
package test;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.awt.Color;
import java.awt.image.BufferedImage;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.jlab.groot.base.ColorPalette;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import domain.objectDetection.FaultObjectClassifier;
import faultrecordreader.FaultObjectDetectionRecordReader;
import faultrecordreader.FaultRecorderScaler;
import faults.FaultNames;
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
	private FaultObjectDetectionRecordReader recordReader;

	public DrawFaults() {
		initialize();
	}

	private void initialize() {
		this.recordReader = new FaultObjectDetectionRecordReader(3, 10, FaultNames.CHANNEL_ONE, true, true, 112, 6, 3,
				56, 3);
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

	public FaultObjectDetectionRecordReader getRecordReader() {
		return this.recordReader;
	}

	public void loadImage(INDArray arr) throws java.lang.Exception {
		double max = (double) arr.maxNumber();
		System.out.println(max + " max from loadImage");
		int rank = arr.rank();
		int rows = arr.size(rank == 3 ? 1 : 2);
		int cols = arr.size(rank == 3 ? 2 : 3);
		int nchannels = arr.size(rank == 3 ? 0 : 1);

		int xLength = cols;// arr.columns();
		int yLength = rows;// arr.rows();

		BufferedImage b = new BufferedImage(rows, cols, BufferedImage.TYPE_INT_RGB);
		ColorPalette palette = new ColorPalette();
		if (nchannels == 1) {
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {

					Color weightColor = palette.getColor3D(arr.getDouble(0, 0, y, x), max, false);

					int red = weightColor.getRed();
					int green = weightColor.getGreen();
					int blue = weightColor.getBlue();
					int rgb = (red * 65536) + (green * 256) + blue;

					b.setRGB(y, cols - x - 1, rgb);

				}
			}
		} else if (nchannels == 3) {
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {

					double red = arr.getDouble(0, 0, y, x);
					double green = arr.getDouble(0, 1, y, x);
					double blue = arr.getDouble(0, 2, y, x);
					double rgb = (red * 65536) + (green * 256) + blue;

					b.setRGB(y, cols - x - 1, (int) rgb);

				}
			}
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}
		CanvasFrame cframe = new CanvasFrame("Loaded Image made me");
		cframe.setTitle("LI Plot");
		cframe.setCanvasSize(800, 600);
		cframe.showImage(b);
	}

	public static void main(String[] args) throws Exception {
		int gridWidth = 6;
		int gridHeight = 112;

		DrawFaults drawFaults = new DrawFaults();

		RecordReaderDataSetIterator test = (RecordReaderDataSetIterator) drawFaults.getDSIterator();
		org.nd4j.linalg.dataset.DataSet ds = test.next();

		// for (String string : labels) {
		// System.out.println(string);
		// }
		// RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)
		// ds.getExampleMetaData().get(0);
		NativeImageLoader imageLoader = new NativeImageLoader();
		CanvasFrame frame = new CanvasFrame("Test");
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

		INDArray features = ds.getFeatures();
		// drawFaults.loadImage(features);
		FaultUtils.draw(features);
		Mat mat = imageLoader.asMat(features);
		Mat convertedMat = new Mat();
		mat.convertTo(convertedMat, 0, 255, 0);
		// mat.convertTo(convertedMat, 6, 255, 0);

		int w = 6;
		int h = 112;
		Mat image = new Mat();

		resize(convertedMat, image, new Size(w, h));

		frame.setTitle(" - test");
		frame.setCanvasSize(w * 200, h * 5);
		frame.showImage(converter.convert(convertedMat));

	}
}

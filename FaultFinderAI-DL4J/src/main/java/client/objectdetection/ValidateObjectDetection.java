/**
 * 
 */
package client.objectdetection;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.CLASObject.ContainerType;
import clasDC.objects.SuperLayer;
import domain.objectDetection.FaultObjectClassifier;
import domain.objectDetection.FaultObjectContainer;
import faultrecordreader.FaultRecorderScaler;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
public class ValidateObjectDetection {

	int height;
	int width;
	int gridHeight;
	int gridwidth;
	int channels;
	private String fileName;
	private DataSetIterator test;
	private FaultObjectClassifier classifier;

	private FaultObjectContainer container;

	public ValidateObjectDetection(String fileName, FaultObjectContainer container) {
		this.fileName = fileName;
		this.container = container;
		initialize();
		try {
			loadClassifier();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			throw new RuntimeException("Couldn't load classifier. Path might be incorrect.", e);
		}
	}

	private void initialize() {
		CLASObject clasObject = container.getClasObject();
		this.height = clasObject.getHeight();
		this.width = clasObject.getWidth();
		this.channels = clasObject.getNchannels();

		RecordReader recordReader = container.getRecordReader();
		FaultRecordScalerStrategy strategy = new MinMaxStrategy();
		this.test = new RecordReaderDataSetIterator.Builder(recordReader, 1).regression(1).maxNumBatches(1)
				.preProcessor(new FaultRecorderScaler(strategy)).build();
	}

	private void loadClassifier() throws IOException {
		this.classifier = new FaultObjectClassifier(fileName);
	}

	public FaultObjectClassifier getClassifier() {
		return this.classifier;
	}

	public DataSetIterator getDSIterator() {
		return this.test;
	}

	public static void main(String[] args) {
		/**
		 * Create a CLASObject for the container
		 */
		CLASObject clasObject = SuperLayer.builder().superlayer(1).nchannels(1).minFaults(1).maxFaults(3)
				.desiredFaults(Stream.of(FaultNames.FUSE_A).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(true).isScaled(false).containerType(ContainerType.OBJ).build();
		/**
		 * FaultObjectContainer contains all the necessaries to run the model
		 */
		FaultObjectContainer container = FaultObjectContainer.builder().clasObject(clasObject).build();

		Set<String> labelSet = new HashSet<>();
		for (FaultNames d : clasObject.getDesiredFaults()) {
			labelSet.add(d.getSaveName());
		}

		List<String> faultLabels = new ArrayList<>(labelSet);
		Collections.sort(faultLabels);

		FaultObjectClassifier classifier;
		String fileName = "models/binary_classifiers/ComputationalGraphModel/SuperLayerSingleFault_FUSEA_15.zip";
		// List<String> labels = train.getLabels();
		ValidateObjectDetection vObjectClassifier = new ValidateObjectDetection(fileName, container);

		ComputationGraph model = container.getModel();
		int gridHeight = container.getGridHeight();
		int gridWidth = container.getGridWidth();

		org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model
				.getOutputLayer(0);
		RecordReaderDataSetIterator test = (RecordReaderDataSetIterator) vObjectClassifier.getDSIterator();

		boolean noFaultFound = true;

		// for (int i = 0; i < 100; i++) {
		while (noFaultFound) {

			org.nd4j.linalg.dataset.DataSet ds = test.next();
			List<String> labels = test.getLabels();
			// System.out.println(ds.getFeatureMatrix().columns() + " " +
			// ds.getFeatureMatrix().rows());

			// for (String string : labels) {
			// System.out.println(string);
			// }
			// RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)
			// ds.getExampleMetaData().get(0);
			OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

			INDArray features = ds.getFeatures();
			INDArray results = model.outputSingle(features);
			List<DetectedObject> objs = yout.getPredictedObjects(results, 0.512);

			if (objs.size() > 1) {
				System.out.println(objs.size() + "  size of objs");
				System.out.println("#################");
				System.out.println("#################");
				noFaultFound = false;
				// try {
				// vObjectClassifier
				// .loadImage(vObjectClassifier.getRecordReader().getFactory().getFeatureVectorAsMatrix());
				// } catch (Exception e) {
				// // TODO Auto-generated catch block
				// e.printStackTrace();
				// }
				FaultUtils.draw(features);
				NativeImageLoader imageLoader = new NativeImageLoader();
				Mat mat = imageLoader.asMat(features);
				Mat convertedMat = new Mat();
				mat.convertTo(convertedMat, CV_8U, 255, 0);
				int w = 112 * 4;
				int h = 6 * 100;
				Mat image = new Mat();

				resize(convertedMat, image, new Size(w, h));
				for (DetectedObject obj : objs) {
					System.out.println(obj.toString() + "  " + (obj.getPredictedClass()) + "  "
							+ faultLabels.get(obj.getPredictedClass()));
					double[] xy1 = obj.getTopLeftXY();
					double[] xy2 = obj.getBottomRightXY();
					String label = labels.get(obj.getPredictedClass());
					int x1 = (int) Math.round((double) w * xy1[0] / (double) gridWidth);
					int y1 = (int) Math.round((double) h * xy1[1] / (double) gridHeight);
					int x2 = (int) Math.round((double) w * xy2[0] / (double) gridWidth);
					int y2 = (int) Math.round((double) h * xy2[1] / (double) gridHeight);
					rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.YELLOW);
					putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
				}
				CanvasFrame frame = new CanvasFrame("Valididate");
				frame.setTitle(" Fault - Valididation");
				frame.setCanvasSize(w, h);
				frame.showImage(converter.convert(image));
				try {
					frame.waitKey();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

}

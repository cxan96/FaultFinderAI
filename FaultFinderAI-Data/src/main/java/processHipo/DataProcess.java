package processHipo;

import java.util.ArrayList;
import java.util.List;

import org.datavec.image.data.Image;
import org.jlab.io.base.DataBank;
import org.jlab.io.base.DataEvent;
import org.jlab.io.hipo.HipoDataSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import strategies.FaultRecordScalerStrategy;
import utils.FaultUtils;

public class DataProcess {
	private HipoDataSource reader = null;
	private List<String> fileList = null;
	private int processN;
	private int counter = 0;
	private FaultDataContainer fContainer = null;

	public DataProcess() {
		init();
	}

	public DataProcess(String... strings) {
		this.fileList = new ArrayList<>();
		createFileList(strings);
		init();
	}

	public DataProcess(List<String> fileList) {
		this(fileList, Integer.MAX_VALUE);
	}

	public DataProcess(List<String> fileList, int processN) {
		this.processN = processN;
		this.fileList = fileList;
		init();
	}

	private void init() {
		this.fContainer = new FaultDataContainer();
	}

	public void setFileList(List<String> list) {
		this.fileList = list;
	}

	private void createFileList(String... strings) {
		for (String string : strings) {
			this.fileList.add(string);
		}
	}

	public void processFile() {
		for (String string : fileList) {
			this.reader = new HipoDataSource();
			this.reader.open(string);
			fillData();
			this.reader.close();
		}
	}

	public void plotData() {
		this.fContainer.plotData();
	}

	public void plotData(int sector, int superLayer) {
		this.fContainer.plotData(sector, superLayer);
	}

	public void plotData(int sector, int superLayer, boolean userChoice) {
		this.fContainer.plotData(sector, superLayer, userChoice);
	}

	private void fillData() {

		while (this.reader.hasEvent() && counter < this.processN) {// && counter
																	// < 400 &&
			// counter < nEvents
			if (counter % 10000 == 0) {
				System.out.println("done " + counter + " events");
			}
			DataEvent event = reader.getNextEvent();
			counter++;
			if (event.hasBank("TimeBasedTrkg::TBHits")) {
				processTBHits(event);
			}
		}
	}

	private void processTBHits(DataEvent event) {
		DataBank bnkHits = event.getBank("TimeBasedTrkg::TBHits");
		for (int i = 0; i < bnkHits.rows(); i++) {

			fContainer.increment(bnkHits.getInt("sector", i), bnkHits.getInt("superlayer", i),
					bnkHits.getInt("wire", i), bnkHits.getInt("layer", i));

		}
	}

	public int[][] getData(int sector, int superLayer) {
		return fContainer.getData(sector, superLayer);
	}

	public INDArray getFeatureVector(int sector, int superLayer) {
		INDArray array = NDArrayUtil.toNDArray(ArrayUtil.flatten(this.getData(sector, superLayer)));

		return array;
	}

	public INDArray getFeatureVector(int sector, int superLayer, FaultRecordScalerStrategy strategy) {
		INDArray array = getFeatureVector(sector, superLayer);
		strategy.normalize(array);
		return array;
	}

	public INDArray getFeatureVectorAsMatrix(int sector, int superLayer) {
		return Nd4j.create(FaultUtils.convertToDouble(this.getData(sector, superLayer)));
	}

	public INDArray getFeatureVectorAsMatrix(int sector, int superLayer, FaultRecordScalerStrategy strategy) {
		INDArray array = Nd4j.create(FaultUtils.convertToDouble(this.getData(sector, superLayer)));
		strategy.normalize(array);

		return array;
	}

	public Image asImageMartix(int sector, int superLayer, int nchannels) {
		return FaultUtils.asImageMatrix(nchannels, getFeatureVectorAsMatrix(sector, superLayer));

	}

	public Image asImageMartix(int sector, int superLayer, int nchannels, FaultRecordScalerStrategy strategy) {
		return FaultUtils.asImageMatrix(nchannels, getFeatureVectorAsMatrix(sector, superLayer, strategy));

	}

}

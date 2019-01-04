package client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import processHipo.DataProcess;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class ClassifyRealData {

	private String dataDir = null;
	private List<String> aList = null;
	private List<FaultNames> fautList = null;

	private DataProcess dataProcess;
	private FaultRecordScalerStrategy strategy;
	private boolean singleModels = false;
	private int nchannels = 1;

	public ClassifyRealData() {
		this.dataDir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";
		this.aList = new ArrayList<>();
		fautList = new ArrayList<>();

		makeList();
		this.dataProcess = new DataProcess(aList);
		this.strategy = new MinMaxStrategy();
	}

	private void makeList() {
		aList.add(dataDir + "out_clas_003923.evio.80.hipo");
		// aList.add(dataDir + "out_clas_003923.evio.8.hipo");

		// aList.add(dataDir + "out_clas_003971.evio.1000.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1001.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1002.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1003.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1004.hipo");
	}

	public void runSingleModels() throws IOException {
		dataProcess.processFile();
		Map<Pair<Integer, Integer>, Pair<List<Fault>, Frame>> dataInfo = new HashMap<>();
		for (int sector = 1; sector < 7; sector++) {
			for (int superlayer = 1; superlayer < 7; superlayer++) {
				INDArray featureArray = dataProcess.asImageMartix(sector, superlayer, nchannels, strategy).getImage();
				DetectFaults dFaults = new DetectFaults(featureArray);

				Pair<List<Fault>, Frame> aPair = dFaults.getListandFrame();

				CanvasFrame canvas = new CanvasFrame("Valididate");
				canvas.setTitle(" sector " + sector + " superlayer " + superlayer);
				canvas.setCanvasSize(448, 450);
				canvas.showImage(aPair.getRight());

				dataInfo.put(Pair.of(sector, superlayer), aPair);
			}
		}
	}

	public void setSingleModel(boolean singleModels) {
		this.singleModels = singleModels;
	}

	public void setNchannels(int nchannels) {
		this.nchannels = nchannels;
	}

	public static void main(String[] args) throws IOException {
		ClassifyRealData cData = new ClassifyRealData();
		cData.setSingleModel(true);
		cData.setNchannels(1);
		cData.runSingleModels();
	}

}

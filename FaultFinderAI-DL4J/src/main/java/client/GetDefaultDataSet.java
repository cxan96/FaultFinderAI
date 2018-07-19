package client;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import processHipo.DataProcess;

public class GetDefaultDataSet {
	public static void main(String[] args) {
		String mkDir = "/Users/michaelkunkel/WORK/CLAS/CLAS12/CLAS12Data/RGACooked/V5b.2.1/";

		String dir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";// DomainUtils.getDataLocation();
		List<String> aList = new ArrayList<>();
		// aList.add(mkDir + "out_clas_003923.evio.80.hipo");
		aList.add(mkDir + "out_clas_003923.evio.8.hipo");

		// aList.add(dir + "out_clas_003923.evio.8.hipo");
		DataProcess dataProcess = new DataProcess(aList, 2000);
		dataProcess.processFile();

		// lets try to normalize by zscore for all the wire for each SL
		List<INDArray> processedArray = new ArrayList<>();
		List<INDArray> scaledArray = new ArrayList<>();

		for (int j = 1; j < 7; j++) { // superlayer first
			List<INDArray> arrays = new ArrayList<>();
			for (int i = 1; i < 7; i++) { // sector
				arrays.add(dataProcess.getFeatureVector(i, j));
			}
			processedArray.add(Nd4j.vstack(arrays));
			scaledArray.add(Nd4j.zeros(112 * 6));
			arrays.clear();
		}
		for (int j = 0; j < 6; j++) {
			INDArray vStack = processedArray.get(j);
			for (int i = 0; i < vStack.columns(); i++) {
				System.out.println(vStack.getColumn(i).meanNumber());
				scaledArray.get(j).putScalar(i, (double) vStack.getColumn(i).meanNumber());
			}
		}

		System.out.println("#########################");
		for (int i = 0; i < scaledArray.get(0).length(); i++) {
			System.out.println(scaledArray.get(0).getDouble(i) + "  " + i);
		}

	}
}

package client;

import java.util.ArrayList;
import java.util.List;

import processHipo.DataProcess;

public class ClassifyRealData {

	public static void main(String[] args) {
		String dir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";
		List<String> aList = new ArrayList<>();
		aList.add(dir + "out_clas_003923.evio.80.hipo");
		// aList.add(dir + "out_clas_003923.evio.8.hipo");
		DataProcess dataProcess = new DataProcess(aList);
		dataProcess.processFile();
		// dataProcess.plotData();
		int[][] someData = dataProcess.getData(1, 1);

	}

}

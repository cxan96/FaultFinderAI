package client;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import processHipo.DataProcess;

public class GetDefaultDataSet {
	public static void main(String[] args) {
		// String mkDir =
		// "/Users/michaelkunkel/WORK/CLAS/CLAS12/CLAS12Data/RGACooked/V5b.2.1/";

		String dir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";//
		// DomainUtils.getDataLocation();
		// String dir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/5b.3.3/";
		List<String> aList = new ArrayList<>();
		aList.add(dir + "out_clas_003923.evio.80.hipo");
		aList.add(dir + "out_clas_003923.evio.8.hipo");
		// aList.add(dir + "out_clas_003971.evio.1000.hipo");
		// aList.add(dir + "out_clas_003971.evio.1001.hipo");
		// aList.add(dir + "out_clas_003971.evio.1002.hipo");
		// aList.add(dir + "out_clas_003971.evio.1003.hipo");
		// aList.add(dir + "out_clas_003971.evio.1004.hipo");

		// aList.add(dir + "out_clas_003923.evio.8.hipo");
		DataProcess dataProcess = new DataProcess(aList);
		dataProcess.processFile();

		// lets try to normalize by zscore for all the wire for each SL
		List<INDArray> processedArray = new ArrayList<>();
		List<INDArray> scaledArray = new ArrayList<>();

		for (int j = 1; j < 7; j++) { // superlayer first
			List<INDArray> arrays = new ArrayList<>();
			for (int i = 1; i < 7; i++) { // sector
				arrays.add(dataProcess.getFeatureVector(i, j));
				// dataProcess.plotData(i, j);
			}
			processedArray.add(Nd4j.vstack(arrays));
			scaledArray.add(Nd4j.zeros(112 * 6));
			arrays.clear();
		}
		for (int j = 0; j < 6; j++) {
			INDArray vStack = processedArray.get(j);
			for (int i = 0; i < vStack.columns(); i++) {
				double fillValue = (double) vStack.getColumn(i).medianNumber();
				if (fillValue == 0.0) {
					int replaceIndex = i + 1;
					if (replaceIndex > 671) {
						replaceIndex = replaceIndex - 2;
					}
					fillValue = (double) vStack.getColumn(replaceIndex).medianNumber();
					if (fillValue == 0.0) {
						replaceIndex = i + 2;
						if (replaceIndex > 671) {
							replaceIndex = replaceIndex - 3;
						}
						fillValue = (double) vStack.getColumn(replaceIndex).medianNumber();
					}
				}
				scaledArray.get(j).putScalar(i, fillValue);
			}
		}
		/**
		 * INDArray tempArray = Nd4j.zeros(6); for (int j = 0; j <
		 * vStack.rows(); j++) { System.out.println(vStack.getDouble(j, i) + " "
		 * + j); tempArray.putScalar(j, vStack.getDouble(j, i)); } new
		 * StandardScore().normalize(tempArray); for (int j = 0; j <
		 * tempArray.columns(); j++) { System.out.println(tempArray.getDouble(i,
		 * j) + " temparray " + j); } System.out.println("vstack mean " +
		 * vStack.getColumn(i).meanNumber()+" vstack mode " +
		 * vStack.getColumn(i).m + " temp mean number" + "" +
		 * tempArray.getRow(0).meanNumber());
		 */

		System.out.println("#########################" + scaledArray.get(0).length());
		// for (int i = 0; i < scaledArray.get(0).length(); i++) {
		// System.out.println(scaledArray.get(0).getDouble(i) + " " + i);
		// }
		for (int init = 0; init < scaledArray.size(); init++) {

			double[][] data = new double[112][6];
			int height = data[0].length;
			int rowPlacer = 0;
			int columnPlacer = 0;
			for (int i = 0; i < scaledArray.get(init).length(); i++) {
				// System.out.println(scaledArray.get(init).getDouble(i) + " " +
				// i);

				double aDub = scaledArray.get(init).getDouble(i);
				if ((i + 1) % height == 0) {
					data[columnPlacer][rowPlacer] = aDub;
					rowPlacer = 0;
					columnPlacer++;
				} else {
					data[columnPlacer][rowPlacer] = aDub;
					rowPlacer++;

				}
			}
			System.out.println("Sector Data for Sector " + init);
			System.out.println(Arrays.deepToString(data));

			TCanvas canvas = new TCanvas("Training Data", 800, 1200);
			H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
			for (int i = 0; i < data[0].length; i++) { // i are the rows
														// (layers)
				for (int j = 0; j < data.length; j++) { // j are the columns
														// (wires)
					hData.setBinContent(j, i, data[j][i]);
				}
			}
			canvas.draw(hData);
		}

	}
}

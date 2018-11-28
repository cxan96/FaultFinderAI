/**
 * 
 */
package client;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

import processHipo.DataProcess;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
public class TestDataProcess {
	public static void main(String[] args) {
		String dir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";
		List<String> aList = new ArrayList<>();
		aList.add(dir + "out_clas_003923.evio.80.hipo");
		// aList.add(dir + "out_clas_003923.evio.8.hipo");
		DataProcess dataProcess = new DataProcess(aList);
		dataProcess.processFile();
		dataProcess.plotData(1, 1);

		FaultUtils.draw(dataProcess.asImageMartix(1, 1, 3));
		FaultRecordScalerStrategy strategy = new MinMaxStrategy();
		FaultUtils.draw(dataProcess.asImageMartix(1, 1, 3, strategy));
		INDArray arr1 = dataProcess.asImageMartix(1, 1, 3, strategy).getImage();
		INDArray arr2 = dataProcess.asImageMartix(1, 1, 3).getImage();

		int rank = arr1.rank();
		int rows = (int) arr1.size(rank == 3 ? 1 : 2);
		int cols = (int) arr1.size(rank == 3 ? 2 : 3);
		int nchannels = (int) arr1.size(rank == 3 ? 0 : 1);

		int rank2 = arr2.rank();
		int rows2 = (int) arr2.size(rank == 3 ? 1 : 2);
		int cols2 = (int) arr2.size(rank == 3 ? 2 : 3);
		int nchannels2 = (int) arr2.size(rank == 3 ? 0 : 1);
		System.out.println(rank + "  " + rows + "  " + cols + "  " + nchannels);
		System.out.println(rank2 + "  " + rows2 + "  " + cols2 + "  " + nchannels2);

		NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
		// INDArray arr3 = dataProcess.asImageMartix(1, 1, 3,
		// preProcessor).getImage();

	}

}

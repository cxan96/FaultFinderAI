package testing;

import java.util.zip.DataFormatException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import utils.FaultUtils;

public class TestArrayConversion {

	public static double[][] convertToDouble(int[][] data) {
		double[][] retValue = new double[data.length][data[0].length];
		for (int i = 0; i < data[0].length; i++) {
			for (int j = 0; j < data.length; j++) {
				retValue[j][i] = (double) data[j][i];
			}
		}
		return retValue;
	}

	public static void main(String[] args) throws DataFormatException {
		int[][] data = FaultUtils.getData(3);
		double[][] newData = convertToDouble(data);
		for (int i = 0; i < newData[0].length; i++) {
			for (int j = 0; j < newData.length; j++) {
				// System.out.println(newData[j][i] + " " + data[j][i]);
			}
		}
		// FaultUtils.draw(newData);
		int[][] d = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
		INDArray ar1 = Nd4j.create(FaultUtils.convertToDouble(d));
		int[][] d2 = { { 6, 5 }, { 4, 3 }, { 2, 1 } };
		INDArray ar2 = Nd4j.create(FaultUtils.convertToDouble(d2));

		INDArray test = Nd4j.concat(0, ar1, ar2);
		int rank = test.rank();
		int rows = test.rows();
		int cols = test.columns();

		System.out.println(rank + "  " + test.columns() + "  " + test.rows());

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				System.out.println(test.getDouble(y, x) + "  " + x + "  " + y);

			}
		}

	}
}

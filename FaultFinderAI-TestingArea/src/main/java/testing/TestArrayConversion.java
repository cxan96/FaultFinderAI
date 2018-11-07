package testing;

import java.util.zip.DataFormatException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

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
		double[][] d = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
		INDArray array = Nd4j.create(17, 10, 20);
		System.out.println(array.size(0));
		INDArray test = Nd4j.create(FaultUtils.convertToDouble(data));
		System.out.println(test.size(0) + "   " + test.shapeInfoToString());

		INDArray ret = test.reshape(ArrayUtil.combine(new int[] { 1 }, test.shape()));
		System.out.println(ret.size(0) + "   " + ret.shapeInfoToString());

	}
}

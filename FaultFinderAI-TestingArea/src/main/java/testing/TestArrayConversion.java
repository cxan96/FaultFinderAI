package testing;

import java.util.zip.DataFormatException;

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
				System.out.println(newData[j][i] + "    " + data[j][i]);
			}
		}
		FaultUtils.draw(newData);
	}
}

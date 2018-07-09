package testing;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.NDArrayUtil;

public class ArrayFun {

	private int[] label;

	public ArrayFun(int[]... coord) {
		makeReducedLabelArray(coord);
	}

	private int getArraySize(int[]... coord) {
		int size = 0;
		for (int[] is : coord) {
			size += is.length;
		}
		return size;
	}

	private void makeReducedLabelArray(int[]... coord) {
		label = new int[getArraySize(coord)];
		int sizePlacer = 0;
		for (int i = 0; i < coord.length; i++) {
			System.arraycopy(coord[i], 0, this.label, sizePlacer, coord[i].length);
			sizePlacer += coord[i].length;
		}
	}

	public int[] getLabel() {
		return this.label;
	}

	public int getLabelSize() {
		return this.label.length;
	}

	public static void main(String[] args) {
		int[] ar1 = { 2, 3, 4 };
		int[] ar2 = { 6, 7, 15 };
		int[] ar3 = { 9, 10, 11, 12 };
		System.out.println("Here");
		ArrayFun aFun = new ArrayFun(ar1, ar2, ar3);
		for (int i : aFun.getLabel()) {
			System.out.print(i + " ");
		}
		System.out.println("\n " + aFun.getLabelSize());
		INDArray array = NDArrayUtil.toNDArray(aFun.getLabel());
		System.out.println(array.maxNumber() + "    max");
		System.out.println(Arrays.toString(array.toDoubleVector()));
		double maxRange = (double) array.maxNumber();
		double minRange = (double) array.minNumber();
		System.out.println("Max = " + maxRange + "  Min = " + minRange);
		// if (minRange != 0.0) {
		// System.out.println(" GLGLKDKJH:");
		// array.subi(minRange); // Offset by minRange
		// }
		System.out.println(Arrays.toString(array.toDoubleVector()) + " Mins subtractd");

		array.divi((maxRange));
		System.out.println(Arrays.toString(array.toDoubleVector()));
		System.out.println(Arrays.toString(array.toDoubleVector()));

	}
}

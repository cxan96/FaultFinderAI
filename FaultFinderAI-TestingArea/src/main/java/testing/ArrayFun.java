package testing;

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
		int[] ar1 = { 1, 2, 3, 4 };
		int[] ar2 = { 6, 7 };
		int[] ar3 = { 9, 10, 11 };
		System.out.println("Here");
		ArrayFun aFun = new ArrayFun(ar1, ar2, ar3);
		for (int i : aFun.getLabel()) {
			System.out.print(i + " ");
		}
		System.out.println("\n " + aFun.getLabelSize());

	}
}

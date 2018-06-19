package testing;

public class FaultArray {

	protected int[] oneFault = { 1, 2, 3, 4, 5 };
	protected int[] twoFault = { 6, 7, 8 };

	public FaultArray() {
	}

	public int[] makeFuseDefaultLabel(int length) {
		int[] deFault = new int[length];
		for (int i = 0; i < deFault.length; i++) {
			deFault[i] = 0;
		}
		return deFault;
	}

	public int[] getOneFault() {
		return oneFault;
	}

	public static void main(String[] args) {
		// FaultArray fArray = new FaultArray();
		// int[] test = fArray.getOneFault();
		//
		// int[] test2 = fArray.makeFuseDefaultLabel(5);
		//
		// test2 = test;
		// for (int i : test2) {
		// System.out.println(i);
		// }

		int[] array1 = { 1, 2, 3 };
		int[] array2 = { 4, 5, 6, 7 };
		int[] array3 = { 8, 9, 10 };

		int[] array4 = new int[array1.length + array2.length + array3.length];

		System.arraycopy(array1, 0, array4, 0, array1.length);
		System.arraycopy(array2, 0, array4, array1.length, array2.length);
		System.arraycopy(array3, 0, array4, array1.length + array2.length, array3.length);

		for (int i = 0; i < array4.length; i++) {
			System.out.print(array4[i] + ", ");
		}

	}
}

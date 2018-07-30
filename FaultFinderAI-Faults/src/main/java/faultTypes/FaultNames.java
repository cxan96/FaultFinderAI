package faultTypes;

/**
 * This enum is used to deal with fault names in a readable manner.
 *
 * Each fault has its designated name and a number that represents its index in
 * the label.
 */
public enum FaultNames {
	PIN_SMALL("Small Pin", 0), PIN_BIG("Big Pin", 1), CHANNEL_ONE("Channel 1", 2), CHANNEL_TWO("Channel 2",
			3), CHANNEL_THREE("Channel 3", 4), CONNECTOR_E("Connector E", 5), CONNECTOR_TREE("Connector Tree",
					6), CONNECTOR_THREE("Connector Three", 7), FUSE_A("Fuse A", 8), FUSE_B("Fuse B", 9), FUSE_C(
							"Fuse C", 10), DEADWIRE("Dead Wire", 11), HOTWIRE("Hot Wire", 12), NOFAULT("No Fault", 13);

	private final String name;
	private final int index;

	FaultNames(String name, int index) {
		this.name = name;
		this.index = index;
	}

	public String toString() {
		return this.name;
	}

	public int getIndex() {
		return this.index;
	}

	public static void main(String args[]) {
		System.out.println(FaultNames.PIN_SMALL.getIndex() + ": " + FaultNames.PIN_SMALL);
		System.out.println(FaultNames.PIN_BIG.getIndex() + ": " + FaultNames.PIN_BIG);
	}
}

package faults;

/**
 * This enum is used to deal with fault names in a readable manner.
 *
 * Each fault has its designated name and a number that represents its index in
 * the label.
 */
public enum FaultNames {
	PIN_SMALL("pin_small", 0, "pin_small"), PIN_BIG("pin_big", 1, "pin_big"), CHANNEL_ONE("Channel 1", 2, "Channel_1"),
	CHANNEL_TWO("Channel 2", 3, "Channel_2"), CHANNEL_THREE("Channel 3", 4, "Channel_3"),
	CONNECTOR_E("Connector E", 5, "Connector_E"), CONNECTOR_TREE("Connector Tree", 6, "Connector_Tree"),
	CONNECTOR_THREE("Connector Three", 7, "Connector_Three"), FUSE_A("Fuse A", 8, "Fuse_A"),
	FUSE_B("Fuse B", 9, "Fuse_B"), FUSE_C("Fuse C", 10, "Fuse_C"), DEADWIRE("deadwire", 11, "Deadwire"),
	HOTWIRE("hotwire", 12, "Hotwire"), NOFAULT("No Fault", 13, "No_Fault");

	private final String name;
	private final int index;
	private final String saveName;

	FaultNames(String name, int index, String saveName) {
		this.name = name;
		this.index = index;
		this.saveName = saveName;
	}

	public String toString() {
		return this.name;
	}

	public int getIndex() {
		return this.index;
	}

	public String getSaveName() {
		return this.saveName;
	}

	public static void main(String args[]) {
		System.out.println(FaultNames.PIN_SMALL.getIndex() + ": " + FaultNames.PIN_SMALL + "  "
				+ FaultNames.PIN_SMALL.getSaveName());
		System.out.println(FaultNames.PIN_BIG.getIndex() + ": " + FaultNames.PIN_BIG);
		System.out.println(FaultNames.CHANNEL_ONE.getIndex() + ": " + FaultNames.CHANNEL_ONE + "  "
				+ FaultNames.CHANNEL_ONE.getSaveName());

		FaultNames aNames = FaultNames.CHANNEL_ONE;
		System.out.println(aNames.getSaveName());
		System.out.println("###########################################");
		for (FaultNames d : FaultNames.values()) {
			System.out.println(d.getSaveName());
		}
	}
}

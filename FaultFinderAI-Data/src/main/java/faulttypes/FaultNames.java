package faulttypes;

/**
 * This enum is used to deal with fault names in a readable manner.
 *
 * Each fault has its designated name and a number
 * that represents its index in the label.
 */
public enum FaultNames {
    PIN_SMALL("Small Pin", 0),
    PIN_BIG("Big Pin", 1),
    CHANNEL_E("Channel E", 2),
    CHANNEL_THREE("Channel Three", 3),
    CHANNEL_TREE("Channel Tree", 4),
    CONNECTOR_A("Connector A", 5),
    CONNECTOR_B("Connector B", 6),
    CONNECTOR_C("Connector C", 7),
    FUSE_E("Fuse E", 8),
    FUSE_THREE("Fuse Three", 9),
    FUSE_TREE("Fuse Tree", 10),
    DEADWIRE("Dead Wire", 11),
    HOTWIRE("Hot Wire", 12),
    NOFAULT("No Fault", 13);
    
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

    public static void main (String args []) {
	System.out.println(FaultNames.PIN_SMALL.getIndex()+": "+FaultNames.PIN_SMALL);
	System.out.println(FaultNames.PIN_BIG.getIndex()+": "+FaultNames.PIN_BIG);
    }
}

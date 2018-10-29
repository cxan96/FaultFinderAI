package faults;

public class FaultCoordinates {
	private final int xMin;
	private final int yMin;
	private final int xMax;
	private final int yMax;
	private final String label;

	public FaultCoordinates(int xMin, int yMin, int xMax, int yMax) {
		this(xMin, yMin, xMax, yMax, null);

	}

	public FaultCoordinates(int xMin, int yMin, int xMax, int yMax, String label) {
		if (xMin > xMax || yMin > yMax) {
			throw new IllegalArgumentException(
					"Invalid input: (xMin,yMin), top left position must have values less than"
							+ " (xMax,yMax) bottom right position. Received: (" + xMin + "," + yMin + "), (" + xMax
							+ "," + yMax + ")");
		}
		this.xMin = xMin;
		this.yMin = yMin;
		this.xMax = xMax;
		this.yMax = yMax;
		this.label = label;

	}

	public int getxMin() {
		return xMin;
	}

	public int getyMin() {
		return yMin;
	}

	public int getxMax() {
		return xMax;
	}

	public int getyMax() {
		return yMax;
	}

	public int[][] getCoordinateArray() {
		return (new int[][] { { this.xMin, this.yMin }, { this.xMax, this.yMax } });
	}

	public double getXCenterPixels() {
		return faultCenter()[0];
	}

	public double getYCenterPixels() {
		return faultCenter()[1];
	}

	public double[] faultCenter() {
		double xCenter = 0;
		double yCenter = 0;
		if (this.xMax == this.xMin) {
			xCenter = (double) this.xMax - 0.5;
		} else {
			xCenter = (double) (this.xMax + this.xMin) * 0.5;
		}
		// for the layers
		if (this.yMax == this.yMin) {
			yCenter = (double) this.yMax - 0.5;
		} else {
			yCenter = (double) (this.yMax) * 0.5;
		}
		return (new double[] { xCenter, yCenter });

	}

	public void printFaultCoordinates() {
		double[] center = faultCenter();
		System.out.println("########################");
		System.out.println(" xMin \t yMin \t xMax \t yMax");
		System.out.println(" " + this.xMin + " \t " + this.yMin + " \t " + this.xMax + " \t " + this.yMax);
		System.out.println(" xCenter \t yCenter ");
		System.out.println(" " + center[0] + " \t " + center[1]);
		System.out.println("########################");
	}

}

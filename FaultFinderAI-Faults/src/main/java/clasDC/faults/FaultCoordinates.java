package clasDC.faults;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

@Setter(AccessLevel.PRIVATE)
@Getter
public class FaultCoordinates {
	private double xMin;
	private double yMin;
	private double xMax;
	private double yMax;
	private String label;

	public FaultCoordinates(double xMin, double yMin, double xMax, double yMax) {
		this(xMin, yMin, xMax, yMax, null);

	}

	public FaultCoordinates(double xMin, double yMin, double xMax, double yMax, String label) {
		if (xMin > xMax || yMin > yMax) {
			throw new IllegalArgumentException(
					"Invalid input: (xMin,yMin), bottom left position must have values less than"
							+ " (xMax,yMax) top right position. Received: (" + xMin + "," + yMin + "), (" + xMax + ","
							+ yMax + ")");
		}
		this.xMin = xMin;
		this.yMin = yMin;
		this.xMax = xMax;
		this.yMax = yMax;
		this.label = label;

	}

	public double[][] getCoordinateArray() {
		return (new double[][] { { this.xMin, this.yMin }, { this.xMax, this.yMax } });
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
			xCenter = (double) this.xMax + 0.5;
		} else {
			xCenter = (double) (this.xMax + this.xMin) * 0.5;
		}
		// for the layers
		if (this.yMax == this.yMin) {
			yCenter = (double) this.yMax + 0.5;
		} else {
			yCenter = (double) (this.yMax + this.yMin) * 0.5;
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
		System.out.println("########################");

	}

	public FaultCoordinates offsetCoordinates(double offset, String axis) {
		if (axis.toLowerCase().equals("x")) {
			System.out.println("X axis modification");
			this.setXMax(this.xMax + offset);
			this.setXMin(this.xMin + offset);

		} else if (axis.toLowerCase().equals("y")) {
			System.out.println("Y axis modification");
			this.setYMax(this.yMax + offset);
			this.setYMin(this.yMin + offset);

		} else {
			throw new IllegalArgumentException("Invalid input: (x or y axis can only be changed");
		}
		return new FaultCoordinates(this.xMin, this.yMin, this.xMax, this.yMax, this.label);

	}

}

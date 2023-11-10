export default class Matrix {

    private static isClose(a: number, b: number, rtol = 1e-05, atol = 1e-08): boolean {
        return Math.abs(a - b) <= atol + rtol * Math.abs(b);
    }

    private static range(n: number, map?: (e: number) => number): number[] {
        return map === void 0
            ? Array.from(Array(n).keys())
            : Array.from(Array(n).keys(), map);
    }

    public static identity(size: number): Matrix {
        return Matrix.diag(size);
    }

    public static diag(size: number, value?: number): Matrix;
    public static diag(values: number[]): Matrix;
    public static diag(values: Matrix): Matrix;
    public static diag(arg0: number | number[] | Matrix, arg1 = 1): Matrix {
        const [size, getValue] =
            typeof arg0 === "number" ? (
                [arg0, () => arg1]

            ) : Array.isArray(arg0) ? (
                [arg0.length, (i: number) => arg0[i]]

            ) : (
                [Math.max(...arg0.shape), (i: number) => arg0.at(i % arg0.size(0), i % arg0.size(1))]
            );

        return new Matrix(size).forCols((_, i, m) => m.replace(i, i, getValue(i)));
    }

    public static fill(size: number, num?: number): Matrix;
    public static fill(rows: number, cols: number, num: number): Matrix;
    public static fill(arg0: number, arg1 = 0, arg2?: number): Matrix {
        const [rows, cols, num] =
            arg2 === void 0 ? (
                [arg0, arg0, arg1]

            ) : (
                [arg0, arg1, arg2]
            );

        return new Matrix(
            new Float64Array(rows * cols).fill(num),
            false,
            [rows, cols],
        );
    }

    public readonly value: Float64Array;
    public readonly trans: boolean;
    public readonly shape: [number, number];

    public constructor(matrix: Matrix);
    public constructor(rows: number, cols?: number);
    public constructor(values: number[][], trans?: boolean);
    public constructor(values: Float64Array, trans: boolean, shape: [number, number]);
    public constructor(arg0: Matrix | number | number[][] | Float64Array, arg1?: number | boolean, arg2?: [number, number]) {
        if (arg0 instanceof Matrix) {
            this.value = new Float64Array(arg0.value);
            this.trans = arg0.trans;
            this.shape = [...arg0.shape];

        } else if (arg0 instanceof Float64Array) {
            if (typeof arg1 !== "boolean") {
                throw new TypeError();
            } else if (!arg2) {
                throw new TypeError();
            }

            this.value = arg0;
            this.trans = arg1;
            this.shape = arg2;

        } else if (Array.isArray(arg0)) {
            this.value = new Float64Array(arg0.flat());
            this.trans = !!arg1;
            this.shape = [arg0.length, arg0[0].length];

        } else if (typeof arg0 === "number") {
            const [rows, cols] =
                typeof arg1 === "number" ? (
                    [arg0, arg1]
                ) : (
                    [arg0, arg0]
                );

            this.value = new Float64Array(rows * cols);
            this.trans = false;
            this.shape = [rows, cols];

        } else {
            throw new Error();
        }
    }

    public get T(): Matrix {
        return new Matrix(this.value, !this.trans, this.shape);
    }

    public get isSquare(): boolean {
        return this.size(0) === this.size(1);
    }

    public get trace(): number {
        let t = 0;
        this.forCols((c, j) => t += c(j));

        return t;
    }

    public size(axis?: number): number {
        return axis === void 0
            ? this.shape[0] * this.shape[1]
            : this.shape[axis ^ +this.trans];
    }

    public index(row: number, col: number): number {
        const [i, j] = this.trans ? [col, row] : [row, col];
        const [r, c] = this.shape;

        return c * ((r + (i % r)) % r) + (c + (j % c)) % c;
    }

    public at(): number;
    public at(row: null | undefined, col?: null | undefined): number;
    public at(row: null | undefined, col: number): ((i: number) => number);
    public at(row: number, col?: null | undefined): ((j: number) => number);
    public at(row: number, col: number): number;
    public at(row?: null | undefined | number, col?: null | undefined | number): number | ((arg: number) => number) {

        return (
            row === null || row === void 0 ? (
                col === null || col === void 0
                    ? this.at(0, 0)
                    : i => this.at(i, col)

            ) : col === null || col === void 0 ? (
                j => this.at(row, j)

            ) : (
                this.value[this.index(row, col)]
            )
        );
    }

    public replace(row: number, col: number, value: number | ((e: number) => number)): Matrix {
        this.value[this.index(row, col)] =
            typeof value === "number"
                ? value
                : value(this.at(row, col));

        return this;
    }

    public allClose(other: Matrix, rtol?: number, atol?: number): Matrix {
        return this.sub(other).pointwise(e => +Matrix.isClose(e, 0, rtol, atol));
    }

    public fix(): Matrix {
        return this.pointwise(e =>
            Matrix.isClose(e, 0) ? (
                0
            ) : Matrix.isClose(e, 1) ? (
                1
            ) : Matrix.isClose(e, -1) ? (
                -1
            ) : (
                e
            )
        );
    }

    public swap(rows: number[][] = [], cols: number[][] = []): Matrix {
        const [rPerm, cPerm] = [0, 1].map(axis => Matrix.identity(this.size(axis)));

        const perm = (ident: Matrix, pattern: number[][]) => {
            let m = ident;

            pattern.forEach(([a, b]) =>
                m = Matrix.identity(m.size(0))
                    .replace(a, b, 1).replace(b, a, 1)
                    .replace(a, a, 0).replace(b, b, 0)
                    .dot(m)
            );

            return m;
        };

        return perm(rPerm, rows).dot(this).dot(perm(cPerm, cols));
    }

    public diag(axis?: number): Matrix {
        return axis === void 0 ? (
            Matrix.diag(this)

        ) : axis ? (
            this.diag(0).T

        ) : (
            new Matrix([Array.from(Array(this.size(0)), (_, i) => this.at(i, i))])
        );
    }

    public partial(rows: number[] | number[][], cols: number[] | number[][], exclude = false): Matrix {
        const select = (selector: number[] | number[][], axis: number) => {
            const indices = [selector]
                .flat()
                .map(idx =>
                    Array.isArray(idx) ? (
                        Matrix.range(
                            (idx[1] ?? this.size(axis) - 1) - idx[0] + 1,
                            e => e + idx[0]
                        )
                    ) : (
                        idx
                    )
                )
                .flat();

            return exclude ? (
                Matrix.range(this.size(axis)).filter(x => !indices.includes(x))

            ) : (
                indices
            );
        };

        const [rIndex, cIndex] = [select(rows, 0), select(cols, 1)];

        return new Matrix(rIndex.map(i => cIndex.map(j => this.at(i, j))));
    }

    public rows(...rows: number[] | number[][]): Matrix {
        return this.partial(rows, [[0]]);
    }

    public cols(...cols: number[] | number[][]): Matrix {
        return this.partial([[0]], cols);
    }

    public subMatrix(rows: number[] | number[][], cols: number[] | number[][]): Matrix {
        return this.partial(rows, cols, true);
    }

    public concat(other: Matrix, axis = 0): Matrix {
        if (axis) {
            return this.T.concat(other.T).T;

        } else {
            const m = new Matrix(this.size(0) + other.size(0), this.size(1));

            this.forPoints((e, i, j) => m.replace(i, j, e));
            other.forPoints((e, i, j) => m.replace(i + this.size(0), j, e));

            return m;
        }
    }

    public forRows(func: (r: (j: number) => number, i: number, m: Matrix) => unknown, from = 0, to = this.size(0) - 1, step = 1): Matrix {
        for (let i = from; i <= to; i += step) {
            func(this.at(i), i, this);
        }

        return this;
    }

    public forCols(func: (c: (i: number) => number, i: number, m: Matrix) => unknown, from = 0, to = this.size(1) - 1, step = 1): Matrix {
        for (let i = from; i <= to; i += step) {
            func(this.at(null, i), i, this);
        }

        return this;
    }

    public forPoints(func: (e: number, i: number, j: number, m: Matrix) => unknown): Matrix {
        return this.forRows((r, i, m) => m.forCols((_, j) => func(r(j), i, j, m)));
    }

    public pointwise(func: (e: number, i: number, j: number, m: Matrix) => number): Matrix {
        return new Matrix(this).forPoints((e, i, j, m) => m.replace(i, j, func(e, i, j, m)));
    }

    public axiswise(func: (a: number, b: number) => number, axis: number | null | undefined = null, initial = 0): Matrix {
        return (
            axis === null || axis === void 0 ? (
                this.axiswise(func, 0, initial).axiswise(func, 1, initial)

            ) : axis === 0 ? (
                this.T.axiswise(func, 1, initial).T

            ) : (
                Matrix.fill(this.size(0), 1, initial).forRows((_, i, m) =>
                    this.forCols(c => m.replace(i, 0, e => func(e, c(i))))
                )
            )
        );
    }

    public round(weakness = 1): Matrix {
        return this.pointwise(e => Math.round(e * weakness) / weakness);
    }

    public binaryOperator(other: Matrix | number[][] | number, func: (a: number, b: number) => number): Matrix {
        const right = other instanceof Matrix
            ? other
            : new Matrix(Array.isArray(other) ? other : [[other]]);

        return this.pointwise((e, i, j) => func(e, right.at(i, j)));
    }

    public add(other: Matrix | number[][] | number): Matrix {
        return this.binaryOperator(other, (a, b) => a + b);
    }

    public sub(other: Matrix | number[][] | number): Matrix {
        return this.binaryOperator(other, (a, b) => a - b);
    }

    public mul(other: Matrix | number[][] | number): Matrix {
        return this.binaryOperator(other, (a, b) => a * b);
    }

    public div(other: Matrix | number[][] | number): Matrix {
        return this.binaryOperator(other, (a, b) => a / b);
    }

    public sum(axis?: number | null | undefined): Matrix {
        return this.axiswise((a, b) => a + b, axis);
    }

    public mean(axis?: number | null | undefined): Matrix {
        const sum = this.sum(axis);

        return sum.div(this.size(0) / sum.size(0) * this.size(1) / sum.size(1));
    }

    public prod(axis?: number | null | undefined): Matrix {
        return this.axiswise((a, b) => a * b, axis, 1);
    }

    public max(axis?: number | null | undefined): Matrix {
        return this.axiswise((a, b) => Math.max(a, b), axis, Number.NEGATIVE_INFINITY);
    }

    public min(axis?: number | null | undefined): Matrix {
        return this.axiswise((a, b) => Math.min(a, b), axis, Number.POSITIVE_INFINITY);
    }

    public dot(other: Matrix): Matrix {
        return new Matrix(this.size(0), other.size(1))
            .forRows((z, k, m) =>
                other.forCols((y, j) =>
                    this.forCols((x, i) =>
                        m.replace(k, j, x(k) * y(i) + z(j))
                    )
                )
            );
    }

    public lu(): [Matrix, Matrix, Matrix, number] {
        let p = Matrix.identity(this.size(0));
        let l = new Matrix(this.size(0));
        let u = new Matrix(this);
        let sign = 1;

        u.forCols((_, j) => {
            if (Matrix.isClose(u.at(j, j), 0)) {
                let idx = j;

                u.forRows((c, i) => {
                    if (Matrix.isClose(c(j), 0)) {
                        idx = i;
                    }
                }, j + 1);

                if (idx === j) {
                    return;
                }

                [p, l, u] = [p, l, u].map(m => m.swap([[idx, j]]));

                sign *= -1;
            }

            u.forRows((_, i) => {
                if (Matrix.isClose(u.at(i, j), 0)) {
                    return;
                }

                const r = u.at(i, j) / u.at(j, j);

                u.forCols((_, k, m) => m.replace(i, k, x => x - m.at(j, k) * r));

                l.replace(i, j, r);
            }, j + 1);
        });

        return [p, l.add(Matrix.identity(this.size(0))), u, sign];
    }

    public det(): number {
        const [, , u, s] = this.lu();

        return u.diag(0).prod().at(0, 0) * s;
    }

    public inv(): Matrix {
        const [p, l, u] = this.lu();

        const echelon = (m: Matrix) =>
            m.concat(Matrix.identity(this.size(0)), 1)
                .forRows((e, i, m) => {
                    const r = e(i);

                    m.forCols((_, j) => m.replace(i, j, z => z / r));
                })
                .forRows((e, i, m) => {
                    m.forRows((_, k) => {
                        const r = e(k);
                        m.forCols((c, j) => m.replace(i, j, z => z - c(k) * r));
                    }, 0, i - 1);
                }, 1);

        const lEchelon = echelon(l).partial([[0]], [[this.size(0)]]);
        const uEchcelon = echelon(u.T).partial([[0]], [[this.size(0)]]).T;

        return uEchcelon.dot(lEchelon).dot(p);
    }

    public toString(): string {
        const s = Math.max(...this.value.map(e => ("" + Math.abs(e)).length));

        return (
            "["
            + Matrix.range(this.size(0))
                .map(i =>
                    "["
                    + Matrix.range(this.size(1))
                        .map(j =>
                            ` ${+this.at(i, j)}${" ".repeat(s)}`
                                .replace(" -", "-")
                                .slice(0, s + 1)
                        ).join(" ")
                    + "]"
                ).join("\n ")
            + "]"
        );
    }

    public print(): Matrix {
        console.log(this.toString());

        return this;
    }
}


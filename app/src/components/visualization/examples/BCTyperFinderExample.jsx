import React from "react";
import ExampleFigure1 from "./bctypefinder/figure/ExampleFigure1.jsx";
import ExampleFigure2 from "./bctypefinder/figure/ExampleFigure2.jsx";
import ExampleFigure3 from "./bctypefinder/figure/ExampleFigure3.jsx";
import ExampleFigure4 from "./bctypefinder/figure/ExampleFigure4.jsx";
import ExampleFigure5 from "./bctypefinder/figure/ExampleFigure5.jsx";

const BCTyperFinderExample = () => {
    return (
        <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-center text-base-content">
                BCtypeFinder — Example Visualization
            </h2>

            <div className="grid gap-6">
                <ExampleFigure1 />
                <ExampleFigure2 />
                <ExampleFigure3 />
                <ExampleFigure4 />
                <ExampleFigure5 />
            </div>
        </div>
    );
};

export default BCTyperFinderExample;

import React from "react";
import ExampleFigure1 from "./cancersubminer/figure/ExampleFigure1.jsx";
import ExampleFigure2 from "./cancersubminer/figure/ExampleFigure2.jsx";
import ExampleFigure3 from "./cancersubminer/figure/ExampleFigure3.jsx";
import ExampleFigure4 from "./cancersubminer/figure/ExampleFigure4.jsx";
import ExampleFigure5 from "./cancersubminer/figure/ExampleFigure5.jsx";

const BCTyperFinderExample = () => {
    return (
        <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-center text-base-content">
                CancerSubminer — Example Visualization
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

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: "development",
  devtool: 'eval-source-map',
  entry: {
    manager: './src/manager.tsx',
    dashboard: './src/dashboard.tsx',
  },
  output: {
    filename: "bundle-[name].js",
    path: path.resolve(__dirname, "dist"),
  },
  module: {
    rules: [
        {
            test: /\.tsx?$/,
            use: 'ts-loader',
            include: [path.resolve(__dirname, 'src')],
        },
        {
          test: /\.css$/i,
          use: ["style-loader", "css-loader"],
        },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js', '.tsx', '.jsx'],
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "src/*.css", to: "[name][ext]" },
      ]
    })
  ],
  experiments: {
    asyncWebAssembly: true
  },
};

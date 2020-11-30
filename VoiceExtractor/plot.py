from plotly.graph_objects import Layout, Scatter, Figure
from plotly.offline import plot


class Plot:
    def __init__(self, x, y, title, xlabel, ylabel, ylim=None):
        self.x = x
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylim = ylim

    def __repr__(self):
        return 'Formatted plot: ' + self.title

    def create(self):
        layout = Layout(paper_bgcolor='rgba(55,55,55,.4)', plot_bgcolor='rgba(36,36,36,.70)', yaxis=dict(range=self.ylim))
        data = Scatter(x=self.x, y=self.y, name="Wykres", opacity=0.1, mode='lines+markers')
        fig = Figure(data=data, layout=layout)
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="white"
            )
        )
        fig.update_xaxes(showgrid=True, gridwidth=.3, gridcolor="rgba(100,100,100,.80)")
        fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor="rgba(100,100,100,.80)")
        return plot(fig, output_type='div')